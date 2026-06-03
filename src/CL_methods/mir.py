"""
This method is an adapted version of the Avalanche library: https://github.com/ContinualAI/avalanche
Licence on the 28/07/2025:

Copyright (c) 2020 ContinualAI.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import torch
import logging
import time
from tqdm import tqdm
from src.experiment_config import AlgorithmParams
from .utils import calculate_val_loss, get_buffer_iterator
from torch.utils.data import DataLoader
from src.telemanom_model import TelemanomModel

def update_temp(model, grad, lr):
    model_copy = copy.deepcopy(model)
    for g, p in zip(grad, model_copy.parameters()):
        if g is not None:
            p.data = p.data - lr * g
    return model_copy


class MIR():
    """
    Maximally Interfered Retrieval plugin,
    Implements the strategy defined in
    "Online Continual Learning with Maximally Interfered Retrieval"
    https://arxiv.org/abs/1908.04742

    This strategy has been designed and tested in the
    Online Setting (OnlineCLScenario). However, it
    can also be used in non-online scenarios
    """

    def __init__(self, Replay_dataloader: DataLoader, device, params: AlgorithmParams):
        """
        mem_size: int       : Fixed memory size
        subsample: int      : Size of the sample from which to look
                              for highest interfering exemplars
        batch_size_mem: int : Size of the batch sampled from
                              the bigger subsample batch
        """
        super().__init__()
        # One ReplayBuffer for past experiences
        self.buffer_len = len(Replay_dataloader)
        self.buffer_dataloader = get_buffer_iterator(Replay_dataloader)

        self.batch_size_mem = params.batch_size_mem
        self.cl_max_steps_per_epoch = params.cl_max_steps_per_epoch
        self.device = device
        self.params = params

    def before_backward(self, loss, model: TelemanomModel):
        """
        Compute the loss of maximum interfered samples
        """

        if self.buffer_len > 0:
            xref, yref = next(self.buffer_dataloader)
            samples_x, samples_y = xref.to(self.device), yref.to(self.device)

            # Perform the temporary update with current data
            grad = torch.autograd.grad(
                loss,
                model.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            model_updated = update_temp(
                model, grad, model.optimizer.param_groups[0]["lr"]
            )
            # Selection of the most interfering samples, no grad required
            # plus we put the model in eval mode so that the additional
            # forward pass don't influence the batch norm statistics
            # strategy.model.eval()
            # model_updated.eval()
            with torch.no_grad():
                _old_red_strategy = model.criterion.reduction
                model.criterion = type(model.criterion)(reduction='none')
                old_output = model(samples_x)
                old_loss = torch.mean(model.criterion(old_output, samples_y), dim=(1, 2))
                new_output = model_updated(samples_x)
                new_loss = torch.mean(model.criterion(new_output, samples_y), dim=(1, 2))
                loss_diff = new_loss - old_loss

                chosen_samples_indexes = torch.argsort(loss_diff)[
                    len(samples_x) - self.batch_size_mem :
                ]

                model.criterion = type(model.criterion)(reduction=_old_red_strategy)
            # strategy.model.train()
            # Choose the samples and add their loss to the current loss
            chosen_samples_x, chosen_samples_y = (
                samples_x[chosen_samples_indexes],
                samples_y[chosen_samples_indexes]
            )
            replay_output = model(chosen_samples_x)
            replay_loss = model.criterion(replay_output, chosen_samples_y)
            loss += replay_loss

    def train(self, exp_dataloader: DataLoader, eval_dataloader: DataLoader, model: TelemanomModel):
        """Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        ckpt_path = model.config.results_path / "best_model.pth"

        torch.save(model.state_dict(), ckpt_path)
        best_val_loss = calculate_val_loss(model=model,
                                 eval_dataloader=eval_dataloader,
                                 device=self.device,
                                 epoch=-1)
        patience_counter = 0
        for epoch in range(self.params.num_epochs_exp):
            #self._before_training_epoch(model)

            self.cl_max_steps_per_epoch = min(len(exp_dataloader), self.params.cl_max_steps_per_epoch)
            start_time = time.time()
            self.training_epoch(model=model, exp_dataloader=exp_dataloader,
                                max_training_steps=self.cl_max_steps_per_epoch, epoch=epoch)
            logging.info(f"CL Epoch {epoch+1}/{self.params.num_epochs_exp} training completed in {time.time() - start_time:.2f} seconds.")
            #self._after_training_epoch(**kwargs)

            # early stopping condition
            avg_val_loss = calculate_val_loss(model=model,
                                 eval_dataloader=eval_dataloader,
                                 device=self.device,
                                 epoch=epoch)
            if best_val_loss - avg_val_loss > model.config.early_stopping_delta and not model.config.fixed_epochs:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logging.info(f"New best validation loss: {best_val_loss:.4f}.")
                torch.save(model.state_dict(), ckpt_path)
            elif model.config.fixed_epochs:
                logging.info(f"New validation loss: {avg_val_loss:.4f}.")
            else:
                patience_counter += 1
                logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{self.params.stopping_patience_exp}.")

            if patience_counter >= self.params.stopping_patience_exp and not model.config.fixed_epochs:
                logging.info(f"Early stopping triggered at epoch {epoch+1}. Last train loss = {avg_val_loss}")
                break
            elif model.config.fixed_epochs and epoch + 1 >= model.config.fixed_num_epochs:
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Reached fixed number of epochs: {model.config.fixed_num_epochs}. Last train loss = {avg_val_loss}")
                break
        model.load_state_dict(torch.load(ckpt_path))
        model.to(model.config.device)
        return epoch + 1 - self.params.stopping_patience_exp, self.cl_max_steps_per_epoch

    def training_epoch(self, model, exp_dataloader, max_training_steps, epoch):
        """Training epoch.

        :param kwargs:
        :return:
        """
        model.train()
        progress_bar = tqdm(exp_dataloader, total=max_training_steps, desc=f"CL Epoch {epoch+1} training")
        for idx, mbatch in enumerate(progress_bar):
            if idx > max_training_steps:
                break

            inputs, targets = mbatch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            #self.before_training_iteration(model)

            model.optimizer.zero_grad()
            #self.loss = self._make_empty_loss()

            # Forward
            #self._before_forward(**kwargs)
            mb_output = model(inputs)
            #self._after_forward(**kwargs)

            # Loss & Backward
            loss = model.criterion(mb_output, targets)

            self.before_backward(loss, model)
            loss.backward()
            #self.after_backward(model)

            # Optimization step
            #self._before_update(**kwargs)
            model.optimizer.step()
            #self._after_update(**kwargs)

            #self._after_training_iteration(**kwargs)
            progress_bar.set_postfix({"loss": f"{loss.cpu().item():.4f}"})