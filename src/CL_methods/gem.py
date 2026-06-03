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
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from src.telemanom_model import TelemanomModel
from src.experiment_config import AlgorithmParams
import time
from tqdm import tqdm
import logging
from .utils import calculate_val_loss, get_buffer_iterator
import qpsolvers

class GEM():
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, Replay_dataloader: DataLoader, device, params: AlgorithmParams):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()
        self.buffer_len = len(Replay_dataloader)
        self.buffer_dataloader = get_buffer_iterator(Replay_dataloader)
        self.memory_strength = params.memory_strength
        self.cl_max_steps_per_epoch = params.cl_max_steps_per_epoch

        self.G: Tensor = torch.empty(0)
        self.device = device
        self.params = params

    def before_training_iteration(self, model: TelemanomModel):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if self.buffer_len > 0:
            G = []
            model.train()
            for _ in range(self.params.num_replay_batches): #this would use all data from all tasks!!
                model.train()
                model.optimizer.zero_grad()
                xref, yref = next(self.buffer_dataloader)
                xref, yref = xref.to(self.device), yref.to(self.device)
                outputs = model(xref)
                loss = model.criterion(outputs, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            (
                                p.grad.flatten().detach() #added detach due to memory leakage
                                if p.grad is not None
                                else torch.zeros(p.numel(), device=self.device)
                            )
                            for p in model.parameters()
                        ],
                        dim=0,
                    )
                )

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, model):
        """
        Project gradient based on reference gradients
        """

        if self.buffer_len > 0:
            g = torch.cat(
                [
                    (
                        p.grad.flatten().detach() #added detach due to memory leakage
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=self.device)
                    )
                    for p in model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(self.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        # solution with old quadprog library, same as the author's implementation
        # v = quadprog.solve_qp(P, q, G, h)[0]
        # using new library qpsolvers
        v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
    
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
            logging.info(f"Epoch {epoch+1}/{self.params.num_epochs_exp} training completed in {time.time() - start_time:.2f} seconds.")
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

            self.before_training_iteration(model)

            model.optimizer.zero_grad()
            #self.loss = self._make_empty_loss()

            # Forward
            #self._before_forward(**kwargs)
            mb_output = model(inputs)
            #self._after_forward(**kwargs)

            # Loss & Backward
            loss = model.criterion(mb_output, targets)

            #self._before_backward(**kwargs)
            loss.backward()
            self.after_backward(model)

            # Optimization step
            #self._before_update(**kwargs)
            model.optimizer.step()
            #self._after_update(**kwargs)

            #self._after_training_iteration(**kwargs)
            progress_bar.set_postfix({"loss": f"{loss.cpu().item():.4f}"})