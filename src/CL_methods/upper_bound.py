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

import torch
from torch.utils.data import DataLoader
from src.telemanom_model import TelemanomModel
from src.experiment_config import AlgorithmParams
import time
from tqdm import tqdm
import logging
from .utils import calculate_val_loss


class UPPER_BOUND():
    """
    Implements the DER Strategy (the DER++ Strategy uses cross entropy which does not make sense in time series forecasting),
    from the "Dark Experience For General Continual Learning"
    paper, Buzzega et. al, https://arxiv.org/abs/2004.07211
    """

    def __init__(self, replay_dataloader: DataLoader, device, params: AlgorithmParams):

        """
        mem_size: int       : Fixed memory size
        subsample: int      : Size of the sample from which to look
                              for highest interfering exemplars
        batch_size_mem: int : Size of the batch sampled from
                              the bigger subsample batch
        """

        self.buffer_len = len(replay_dataloader)
        self.buffer_dataloader = replay_dataloader
  
        self.cl_max_steps_per_epoch = params.cl_max_steps_per_epoch

        self.device = device
        self.params = params

    def train(self, eval_dataloader: DataLoader, model: TelemanomModel):
        """Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        model.reset_weights() # reinitializes the weights to train a new model

        best_val_loss = calculate_val_loss(model=model, 
                                 eval_dataloader=eval_dataloader, 
                                 device=self.device, 
                                 epoch=-1)
        patience_counter = 0
        for epoch in range(self.params.num_epochs_exp):
            #self._before_training_epoch(model)

            self.cl_max_steps_per_epoch = min(self.buffer_len, self.params.cl_max_steps_per_epoch)
            start_time = time.time()
            self.training_epoch(model=model,
                                max_training_steps=self.cl_max_steps_per_epoch, epoch=epoch)
            logging.info(f"CL Epoch {epoch+1}/{self.params.num_epochs_exp} training completed in {time.time() - start_time:.2f} seconds.")
            #self._after_training_epoch(**kwargs)

            # early stopping condition
            avg_val_loss = calculate_val_loss(model=model, 
                                 eval_dataloader=eval_dataloader, 
                                 device=self.device, 
                                 epoch=epoch)
            if best_val_loss - avg_val_loss > model.config.early_stopping_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logging.info(f"New best validation loss: {best_val_loss:.4f}.")
                torch.save(model.state_dict(), model.config.results_path / "best_model.pth")
            else:
                patience_counter += 1
                logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{self.params.stopping_patience_exp}.")

            if patience_counter >= self.params.stopping_patience_exp:
                logging.info(f"Early stopping triggered at epoch {epoch+1}. Last train loss = {avg_val_loss}")
                break
        model.load_state_dict(torch.load(model.config.results_path / "best_model.pth"))
        model.to(model.config.device)
        return epoch + 1 - self.params.stopping_patience_exp, self.cl_max_steps_per_epoch

    def training_epoch(self, model, max_training_steps, epoch):
        """Training epoch.
        """
        model.train()
        progress_bar = tqdm(self.buffer_dataloader, total=max_training_steps, desc=f"CL Epoch {epoch+1} training")
        for idx, mbatch in enumerate(progress_bar):
            if idx > max_training_steps:
                break

            inputs, targets = mbatch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            model.optimizer.zero_grad()

            mb_output = model(inputs)

            loss = model.criterion(mb_output, targets)
            loss.backward()

            model.optimizer.step()

            progress_bar.set_postfix({"loss": f"{loss.cpu().item():.4f}"})