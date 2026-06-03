import numpy as np
import torch
import logging
from typing import Tuple, List

from ..dataset import ReplayBuffer




class Variance_Greedy(ReplayBuffer):
    """
    This class instanciates an Episode Balanced Buffer from a Replay Buffer
    (Similar to a task balanced buffer but in time-series episodes)
    it will reduce the number of examples in the self.indice list but never access the actual datapoints in self.data_segments

    This class can be called like this: replay_buffer = EpisodeBalancedReplayBuffer(replay_buffer)
    """
    def __init__(self, old_replay_buffer: ReplayBuffer, model_ref):
        super(Variance_Greedy, self).__init__(
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
        self.model = model_ref
        self.device = self.config.device
        self.var_samples = self.config.method_params.var_samples

        new_indices = self.indices.copy()
        self.indices = []
        with torch.no_grad():
            self.add_data(new_indices) # initializes the indices of the first episode

    def choose_indices(self, error_values: torch.Tensor, indices_of_episode: List[Tuple[int, int]], k_keep: int, k_prune: int) -> List[Tuple[int, int]]:
        """
        This function chooses the indices of the episode based on the error values.
        It prunes the highest error values according to self.prune_frac and returns the remaining indices.
        """
        # Just get the top (k_prune + k_keep) positions, unsorted for speed
        _, top_pos = torch.topk(error_values, k_prune + k_keep, largest=True, sorted=False)

        # Now sort only the remaining k_keep
        if k_prune>0:
            top_pos = top_pos[torch.topk(error_values[top_pos], k_keep, largest=False, sorted=True).indices]

        return [indices_of_episode[pos.item()] for pos in top_pos]

    def add_data(self, total_new_indices):
        """
        This function saves the randomly chosen corresponding indices of each episode.
        It updates self.indices to ensure balance.
        Assumes rebalancing happens after inputting each new episode
        """
        logging.info("--- using replay strategy: variance greedy ---") # here stood loss greedy wrongly!
        total_episodes = len(self.episode_segments)
        data_per_episodes = self.config.method_params.buffer_size//total_episodes
        self.episode_indices.append(total_new_indices)  # Append the new indices to the existing list

        self.model.train()
        _old_red_strategy = self.model.criterion.reduction #
        self.model.criterion = type(self.model.criterion)(reduction='none')

        for episode in range(total_episodes):
            variance_values = torch.zeros(len(self.episode_indices[episode]), device=self.device)
            for start_batch in range(0, len(self.episode_indices[episode]), self.config.val_batch_size):

                end_batch = min(start_batch+self.config.val_batch_size, len(self.episode_indices[episode]))
                batch_idxs = total_new_indices[start_batch:end_batch]
                mb_x, mb_y = self.getbatch(batch_idxs)
                mb_x, mb_y = mb_x.to(self.device), mb_y.to(self.device)

                all_predictions = torch.zeros((self.var_samples, mb_y.size(0), mb_y.size(1), mb_y.size(2)), device=self.device)
                for i in range(self.var_samples):
                    preds = self.model(mb_x)
                    all_predictions[i] = preds
                variance_values[start_batch:end_batch] = torch.std(all_predictions, dim=(0, 2, 3), unbiased=True)
            
            self.episode_indices[episode] = self.choose_indices(
                variance_values, 
                self.episode_indices[episode], 
                k_keep=data_per_episodes,
                k_prune = 0
                )
            
        self.indices = [idx_tuple for episode_list in self.episode_indices for idx_tuple in episode_list] # adds all indices to the indice list

        # resetting the model
        self.model.criterion = type(self.model.criterion)(reduction=_old_red_strategy)
        self.model.train()

    def add_processed_data(self, new_train: List[np.ndarray], indices: List[Tuple[int, int]], old_predictions: List[np.ndarray] = []):

            if len(indices)<1:
                logging.warning("no data to add to the replay buffer")

            indice_start = len(self.data_segments)
            self.data_segments += [arr.copy() for arr in new_train]
            self.anomalies += [np.zeros(seg.shape[0]) for seg in new_train]
            self.old_preds += [pred.copy() for pred in old_predictions]

            # filtering the number of data indices randomly
            self.episode_segments.append((self.episode_segments[-1][1]+1, len(self.data_segments)-1))
            total_new_indices = [(indice_start+i, j) for i, j in indices]
            with torch.no_grad():
                self.add_data(total_new_indices)