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
"""GSSPlugin replay plugin.

Code adapted from the repository:
https://github.com/RaptorMai/online-continual-learning
Handles an external memory fulled with samples selected
using the Greedy approach of GSS algorithm.
`before_forward` callback is used to process the current
sample and estimate a score.
"""

import numpy as np
import torch
import logging

from typing import Tuple, List
from ..dataset import ReplayBuffer

class GSS_Greedy(ReplayBuffer):
    """
    This class instanciates an Episode Balanced Buffer from a Replay Buffer
    (Similar to a task balanced buffer but in time-series episodes)
    it will reduce the number of examples in the self.indice list but never access the actual datapoints in self.data_segments

    This class can be called like this: replay_buffer = EpisodeBalancedReplayBuffer(replay_buffer)
    """
    def __init__(self, old_replay_buffer: ReplayBuffer, model_ref):
        super(GSS_Greedy, self).__init__(
            data_segments = old_replay_buffer.data_segments,
            mean = old_replay_buffer.mean.cpu().numpy(),
            std = old_replay_buffer.std.cpu().numpy(),
            config = old_replay_buffer.config,
            anomalies = old_replay_buffer.anomalies,
            output_old_preds = old_replay_buffer.output_old_preds,
            old_preds = old_replay_buffer.old_preds
        )
        self.device = self.config.device
        self.buffer_size = self.config.method_params.buffer_size
        self.buffer_score = torch.FloatTensor(self.buffer_size).fill_(0).to(self.device) # saves the scores of the indices
        self.buffer_mem_strength = self.config.method_params.buffer_mem_strength # defines how many memory samples are used
        self.model = model_ref

        new_indices = self.indices.copy()
        self.indices = []
        with torch.enable_grad():
            self.add_data(new_indices) # initializes the indices of the first episode

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return sim

    def get_grad_vector(self, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.zeros(sum(grad_dims), device=self.device)
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads

    def get_batch_sim(self, grad_dims, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            batch_x: current batch x
            batch_y: current batch y
        Returns: score of current batch, gradient from memory subsets
        """
        mem_grads = self.get_rand_mem_grads(grad_dims, len(batch_x))
        self.model.zero_grad()
        loss = self.model.criterion(self.model(batch_x), batch_y)
        loss.backward()
        batch_grad = self.get_grad_vector(
            self.model.parameters, grad_dims
        ).unsqueeze(0)
        
        batch_sim = max(self.cosine_similarity(mem_grads, batch_grad))
        return batch_sim, mem_grads

    def get_rand_mem_grads(self, grad_dims, gss_batch_size):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
        Returns: gradient from memory subsets
        """
        temp_gss_batch_size = min(gss_batch_size, len(self.indices))
        num_mem_subs = min(
            self.buffer_mem_strength, len(self.indices) // gss_batch_size
        )
        mem_grads = torch.zeros(
            num_mem_subs,
            sum(grad_dims),
            dtype=torch.float32,
            device=self.device,
        )
        shuffeled_inds = torch.randperm(
            len(self.indices), device=self.device
        )
        for i in range(num_mem_subs):
            random_batch_inds = shuffeled_inds[
                i * temp_gss_batch_size : i * temp_gss_batch_size + temp_gss_batch_size
            ]
            batch_idxs = [self.indices[i] for i in random_batch_inds]
            batch_x, batch_y = self.getbatch(batch_idxs)
            self.model.zero_grad()
           
            loss = self.model.criterion(self.model(batch_x.to(self.device)), batch_y.to(self.device))
    
            loss.backward()
            mem_grads[i].data.copy_(
                self.get_grad_vector(self.model.parameters, grad_dims)
            )
        
        return mem_grads

    def get_each_batch_sample_sim(
        self, grad_dims, mem_grads, batch_x, batch_y
    ):
        """
        Args:
            buffer: memory buffer
            grad_dims: gradient dimensions
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = torch.zeros(batch_x.size(0), device=self.device)
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):
            self.model.zero_grad()
            ptloss = self.model.criterion(
                self.model.forward(x.unsqueeze(0)), y.unsqueeze(0)
            )
            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine
            # similarity
            this_grad = self.get_grad_vector(
                self.model.parameters, grad_dims
            ).unsqueeze(0)
            cosine_sim[i] = max(self.cosine_similarity(mem_grads, this_grad))
        
        return cosine_sim

    def add_processed_data(self, new_train: List[np.ndarray], indices: List[Tuple[int, int]], old_predictions: List[np.ndarray] = []):

            if len(indices)<1:
                logging.warning("no data to add to the replay buffer")

            indice_start = len(self.data_segments)
            self.data_segments += [arr.copy() for arr in new_train]
            self.anomalies += [np.zeros(seg.shape[0]) for seg in new_train]
            self.old_preds += [pred.copy() for pred in old_predictions]

            # filtering the number of data indices randomly
            total_new_indices = [(indice_start+i, j) for i, j in indices]
            with torch.enable_grad():
                self.add_data(total_new_indices)

    def add_data(self, total_new_indices):
        """
        After every forward this function select sample to fill
        the memory buffer based on cosine similarity
        """
        num_new_indices = 0

        self.model.train()  # make sure RNNs get training=True
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout,
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d)):
                m.eval()      # keep these deterministic

        # loop over all new indices
        for start_batch in range(0, len(total_new_indices), self.config.batch_size):

            end_batch = min(start_batch+self.config.batch_size, len(total_new_indices))
            batch_idxs = total_new_indices[start_batch:end_batch]
            mb_x, mb_y = self.getbatch(batch_idxs)
            mb_x, mb_y = mb_x.to(self.device), mb_y.to(self.device)
            
            # Compute the gradient dimension
            grad_dims = []
            for param in self.model.parameters():
                grad_dims.append(param.data.numel())
            
            place_left =  self.buffer_size - len(self.indices)
            if place_left <= 0:  # buffer full
                batch_sim, mem_grads = self.get_batch_sim(
                    grad_dims,
                    batch_x=mb_x,
                    batch_y=mb_y,
                )

                if batch_sim < 0:
                    logging.info(f"start of accepted batch: {start_batch}, with sim: {batch_sim}")
                    buffer_score = self.buffer_score[
                        : len(self.indices)
                    ].cpu()

                    buffer_sim = (buffer_score - torch.min(buffer_score)) / (
                        (torch.max(buffer_score) - torch.min(buffer_score)) + 0.01
                    )

                    # draw candidates for replacement from the buffer
                    index = torch.multinomial(
                        buffer_sim, mb_x.size(0), replacement=False
                    ).to(self.device)

                    # estimate the similarity of each sample in the received batch
                    # to the randomly drawn samples from the buffer.
                    batch_item_sim = self.get_each_batch_sample_sim(
                        grad_dims, mem_grads, mb_x, mb_y
                    )

                    # normalize to [0,1]
                    scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)
                    buffer_repl_batch_sim = ((self.buffer_score[index] + 1) / 2).unsqueeze(
                        1
                    )
                    # draw an event to decide on replacement decision
                    outcome = torch.multinomial(
                        torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1),
                        1,
                        replacement=False,
                    )
                    # replace samples with outcome =1
                    added_indx = torch.arange(
                        end=batch_item_sim.size(0), device=self.device
                    )
                    sub_index = outcome.squeeze(1).bool()

                    dst_list = index[sub_index].tolist()            # positions in the buffer to replace
                    src_list = added_indx[sub_index].tolist()       # positions in the incoming mini-batch
                    for d, s in zip(dst_list, src_list):
                        self.indices[d] = batch_idxs[s] 

                    self.buffer_score[index[sub_index]] = batch_item_sim[
                        added_indx[sub_index]
                    ].clone()
                    num_new_indices += sum(sub_index)
            else:
                offset = min(place_left, mb_x.size(0))
                updated_mb_x = mb_x[:offset]
                updated_mb_y = mb_y[:offset]

                # first buffer insertion
                if len(self.indices) == 0:
                    batch_sample_memory_cos = torch.zeros(updated_mb_x.size(0)) + 0.1
                else:
                    # draw random samples from buffer
                    mem_grads = self.get_rand_mem_grads(
                        grad_dims=grad_dims,
                        gss_batch_size=len(mb_x),
                    )
                    # estimate a score for each added sample
                    batch_sample_memory_cos = self.get_each_batch_sample_sim(
                        grad_dims, mem_grads, updated_mb_x, updated_mb_y
                    )
                curr_idx = len(self.indices)
                self.indices.extend(batch_idxs[:offset])
                self.buffer_score[curr_idx : curr_idx + offset].data.copy_(
                    batch_sample_memory_cos
                )

        self.model.train()
        logging.info(f"Number of changed items: {num_new_indices}")