"""
This method is an adapted version of the Avalanche library: https://github.com/ContinualAI/avalanche
(Latent-greedy is adapted from GSS-greedy from the Avalanche library)
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

class Latent_Greedy(ReplayBuffer):
    """
    This class instanciates an Episode Balanced Buffer from a Replay Buffer
    (Similar to a task balanced buffer but in time-series episodes)
    it will reduce the number of examples in the self.indice list but never access the actual datapoints in self.data_segments

    This class can be called like this: replay_buffer = EpisodeBalancedReplayBuffer(replay_buffer)
    """
    def __init__(self, old_replay_buffer: ReplayBuffer, model_ref):
        super(Latent_Greedy, self).__init__(
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
        with torch.no_grad():
            self.add_data(new_indices) # initializes the indices of the first episode

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)

        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        sim = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return sim

    def get_batch_sim(self, latent_dim, batch_x, batch_y):
        """
        Args:
            buffer: memory buffer
            latent_dim: latent dimension
            batch_x: current batch x
            batch_y: current batch y
        Returns: score of current batch, gradient from memory subsets
        """
        mem_latent = self.get_rand_mem_grads(latent_dim, len(batch_x))

        batch_latent = self.model.get_last_latent(batch_x).mean(dim=0).unsqueeze(0)

        batch_sim = max(self.cosine_similarity(mem_latent, batch_latent))
        return batch_sim, mem_latent

    def get_rand_mem_grads(self, latent_dim, gss_batch_size):
        """
        Args:
            buffer: memory buffer
            latent_dim: latent dimension
        Returns: gradient from memory subsets
        """
        temp_gss_batch_size = min(gss_batch_size, len(self.indices))
        num_mem_subs = min(
            self.buffer_mem_strength, len(self.indices) // gss_batch_size
        )
        mem_grads = torch.zeros(
            num_mem_subs,
            latent_dim,
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
            
            last_latent = self.model.get_last_latent(batch_x.to(self.device)).mean(dim=0)

            mem_grads[i].data.copy_(
                last_latent
            )
        return mem_grads

    def get_each_batch_sample_sim(
        self, latent_dim, mem_grads, batch_x, batch_y
    ):
        """
        Args:
            buffer: memory buffer
            latent_dim: latent dimension
            mem_grads: gradient from memory subsets
            batch_x: batch images
            batch_y: batch labels
        Returns: score of each sample from current batch
        """
        cosine_sim = torch.zeros(batch_x.size(0), device=self.device)
        
        ptloss = self.model.get_last_latent(batch_x)

        for i in range(batch_x.size(0)):
            # add the new grad to the memory grads and add it is cosine
            # similarity
            cosine_sim[i] = max(self.cosine_similarity(mem_grads, ptloss[i].unsqueeze(0)))
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
            with torch.no_grad():
                self.add_data(total_new_indices)

    def add_data(self, total_new_indices):
        """
        After every forward this function select sample to fill
        the memory buffer based on cosine similarity
        """

        self.model.eval()  # set model to eval mode

        # loop over all new indices
        for start_batch in range(0, len(total_new_indices), self.config.batch_size):

            end_batch = min(start_batch+self.config.batch_size, len(total_new_indices))
            batch_idxs = total_new_indices[start_batch:end_batch]
            mb_x, mb_y = self.getbatch(batch_idxs)
            mb_x, mb_y = mb_x.to(self.device), mb_y.to(self.device)
            # Compute the gradient dimension
            latent_dim = self.model.config.lstm_layers[1]

            place_left =  self.buffer_size - len(self.indices)
            if place_left <= 0:  # buffer full
                batch_sim, mem_grads = self.get_batch_sim(
                    latent_dim,
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
                        latent_dim, mem_grads, mb_x, mb_y
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
                        latent_dim=latent_dim,
                        gss_batch_size=len(mb_x),
                    )
                    # estimate a score for each added sample
                    batch_sample_memory_cos = self.get_each_batch_sample_sim(
                        latent_dim, mem_grads, updated_mb_x, updated_mb_y
                    )
                curr_idx = len(self.indices)
                self.indices.extend(batch_idxs[:offset])
                self.buffer_score[curr_idx : curr_idx + offset].data.copy_(
                    batch_sample_memory_cos
                )

        self.model.train()