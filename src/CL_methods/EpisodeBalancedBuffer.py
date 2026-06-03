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
import numpy as np
import logging

from typing import Tuple, List

import random
from ..dataset import ReplayBuffer


class EpisodeBalancedBuffer(ReplayBuffer):
    """
    This class instanciates an Episode Balanced Buffer from a Replay Buffer
    (Similar to a task balanced buffer but in time-series episodes)
    it will reduce the number of examples in the self.indice list but never access the actual datapoints in self.data_segments

    This class can be called like this: replay_buffer = EpisodeBalancedReplayBuffer(replay_buffer)
    """
    def __init__(self, old_replay_buffer: ReplayBuffer):
        super(EpisodeBalancedBuffer, self).__init__(
            data_segments = old_replay_buffer.data_segments,
            mean = old_replay_buffer.mean.cpu().numpy(),
            std = old_replay_buffer.std.cpu().numpy(),
            config = old_replay_buffer.config,
            anomalies = old_replay_buffer.anomalies,
            output_old_preds = old_replay_buffer.output_old_preds,
            old_preds = old_replay_buffer.old_preds
        )

        self.buffer_size = self.config.method_params.buffer_size
        self.old_last_segment = -1 # first (next) idx is 0
        self.episode_segments = [(0, len(self.data_segments)-1)] #keeps track of which indices belongs to which episode
        self.episode_indices = []

        self.rebalance_replay_buffer() # initializes the indices of the first episode

    def rebalance_replay_buffer(self):
        """
        This function saves the randomly chosen corresponding indices of each episode.
        It updates self.indices to ensure balance.
        Assumes rebalancing happens after inputting each new episode
        """
        logging.info("--- using replay strategy: episode balanced ---")
        # init first episode
        if len(self.episode_indices) == 0:
            if len(self.indices) > self.config.method_params.buffer_size:
                self.episode_indices.append(random.sample(self.indices.copy(), k=self.buffer_size))
            else:
                self.episode_indices.append(self.indices)
            

        total_episodes = len(self.episode_segments)
        data_per_episodes = self.config.method_params.buffer_size//total_episodes
        for episode in range(total_episodes):
            # reduce the datapoints, if episode is already initialized
            if len(self.episode_indices[episode]) > data_per_episodes:
                self.episode_indices[episode] = random.sample(self.episode_indices[episode], k=data_per_episodes)
        
        self.indices = [idx_tuple for episode_list in self.episode_indices for idx_tuple in episode_list] # adds all indices to the indice list

    def add_processed_data(self, new_train: List[np.ndarray], indices: List[Tuple[int, int]], old_predictions: List[np.ndarray] = []):

            if len(indices)<1:
                logging.warning("no data to add to the replay buffer")

            indice_start = len(self.data_segments)
            self.data_segments += [arr.copy() for arr in new_train]
            self.anomalies += [np.zeros(seg.shape[0]) for seg in new_train]
            self.old_preds += [pred.copy() for pred in old_predictions]

            # filtering the number of data indices randomly
            data_per_episodes = self.config.method_params.buffer_size//(len(self.episode_indices)+1)
            total_new_indices = [(indice_start+i, j) for i, j in indices]
            self.episode_indices.append(random.sample(total_new_indices, k=data_per_episodes))
            self.episode_segments.append((self.episode_segments[-1][1]+1, len(self.data_segments)-1))
            self.rebalance_replay_buffer()