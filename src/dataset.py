# FILE: dataset.py

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from scipy.stats import skew
from typing import Tuple, List, Optional, Union
import logging
from src.experiment_config import ExperimentConfig
from tqdm import tqdm
import pickle
import hashlib
import json
from src.general_utils import get_cache_path

PREPROCESSED_DATA_FOLDER = Path(__file__).parent.parent / "data/preprocessed"
CACHE_DATA_FOLDER = Path(__file__).parent.parent / "data/cache"

def get_split_timestamp(index: pd.DatetimeIndex, validation_split_ratio: float) -> pd.Timestamp:
    """
    Calculates a timestamp to split a time series index for validation.

    Args:
        index (pd.DatetimeIndex): The time series index to be split.
        validation_split_ratio (float): The fraction of data to be used for validation.

    Returns:
        pd.Timestamp: The timestamp at which to split the data.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Input 'index' must be a pandas DatetimeIndex.")
    if not (0 <= validation_split_ratio <= 1):
        raise ValueError("validation_split_ratio must be between 0 and 1.")

    start_time = index.min()
    end_time = index.max()
    duration = end_time - start_time
    train_duration = duration * (1 - validation_split_ratio)
    split_timestamp = start_time + train_duration
    return split_timestamp

class TelemanomDataset(Dataset):
    """
    A PyTorch Dataset class to load and prepare time series data specifically
    for the Telemanom-ESA model.

    This class handles:
    - Creating sliding window samples of (input_window, target_window).

    The primary way to use this class is via the `from_config` classmethod,
    which performs all necessary preparation and returns configured Dataset
    instances for training and validation.
    """

    def __init__(self,
                 data_segments: List[np.ndarray],
                 mean: np.ndarray | List[np.ndarray],
                 std: np.ndarray | List[np.ndarray],
                 config: ExperimentConfig,
                 anomalies: Optional[List[np.ndarray]] = None,
                 output_anomalies: bool = False,
                 dates: Optional[List[pd.Timestamp]] = None
                 ):
        super(TelemanomDataset, self).__init__()
        """
        Private constructor. Use `from_config` to create instances.
        """
        self.data_segments = data_segments
        if isinstance(mean, list):
            self.mean = [torch.from_numpy(m.astype(np.float32)) for m in mean]
        else:
            self.mean = torch.from_numpy(mean.astype(np.float32))
        if isinstance(std, list):
            self.std = [torch.from_numpy(s.astype(np.float32)) for s in std]
        else:
            self.std = torch.from_numpy(std.astype(np.float32))
        self.config = config
        self.window_size = config.window_size
        self.prediction_window_size = config.prediction_window_size
        self.output_anomalies = output_anomalies
        self.dates = dates

        if anomalies is None:
            # For train/val sets, anomalies are pruned, so we have all zeros.
            self.anomalies = [np.zeros(seg.shape[0]) for seg in self.data_segments]
        else:
            self.anomalies = anomalies

        # A single sample is a window of size `window_size` followed by a
        # prediction window of size `prediction_window_size`.
        self.sample_len = self.window_size + self.prediction_window_size

        # Create a flat index map over all valid windows in all segments
        self.indices = []
        for seg_idx, segment in enumerate(self.data_segments):
            if len(segment) >= self.sample_len:
                num_windows = len(segment) - self.sample_len + 1
                # Store as (segment_index, start_position_in_segment)
                self.indices.extend([(seg_idx, i) for i in range(num_windows)])

    @classmethod
    def from_config(cls,
                   data_folder_path: str = PREPROCESSED_DATA_FOLDER,
                   config: ExperimentConfig = ExperimentConfig(),
                   include_test: bool = False,
                   ) -> Union[Tuple, List]:
        """
        Factory method to create train/validation and optionally test dataset instances.
        This is an abstract method that should be implemented by subclasses.

        Args:
            data_folder_path (str): Path to the root data folder.
            config (ExperimentConfig): The configuration object for the experiment.
            include_test (bool): If True, loads test data in addition to train/val data.

        Returns:
            If include_test is False:
                A tuple of (train_dataset, validation_dataset).
                If num_clients > 1, a list of (train_dataset, validation_dataset) tuples.
            If include_test is True:
                A tuple of (train_dataset, validation_dataset, test_dataset).
                If num_clients > 1, a list of (train_dataset, validation_dataset, test_dataset) tuples.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    @classmethod
    def create_or_load(cls,
                       config: ExperimentConfig,
                       loader_func,
                       **kwargs):
        """
        Factory method to create a dataset, using a cache if available.

        Args:
            config (ExperimentConfig): The configuration object for the experiment.
            loader_func (callable): The function to call to load/create the data if not cached.
            **kwargs: Additional arguments for the loader function.

        Returns:
            The loaded or created dataset.
        """
        if not config.use_cache:
            return loader_func(config=config, **kwargs)
        
        base_path = CACHE_DATA_FOLDER / "telemanom"
        base_path.mkdir(parents=True, exist_ok=True)

        cache_path = get_cache_path(
            base_path=base_path,
            config=config,
            prefix=""
        )

        if cache_path.exists():
            logging.info(f"Loading cached dataset from {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                logging.warning(f"Could not load cache file {cache_path}: {e}. Recreating dataset.")

        dataset = loader_func(config=config, **kwargs)
        
        logging.info(f"Caching dataset to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

    def __len__(self) -> int:
        """Returns the total number of valid sliding windows."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single standardized (input_window, target_window, anomaly_labels) pair.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        # Map the flat index to the correct segment and position
        segment_idx, start_pos = self.indices[idx]
        segment = self.data_segments[segment_idx]
        anomaly_segment = self.anomalies[segment_idx]

        # Extract the full sample slice
        sample_slice = segment[start_pos : start_pos + self.sample_len]
        anomaly_slice = anomaly_segment[start_pos : start_pos + self.sample_len]

        # Split into input and target windows
        input_window = sample_slice[:self.window_size]
        target_window = sample_slice[self.window_size:]
        input_anomalies = anomaly_slice[:self.window_size]
        
        # Convert to tensor
        input_tensor = torch.from_numpy(input_window)
        target_tensor = torch.from_numpy(target_window)
        input_anomalies_tensor = torch.from_numpy(input_anomalies).long()

        # Standardize the data
        input_tensor = self.standardize(input_tensor, idx)
        target_tensor = self.standardize(target_tensor, idx)

        if self.output_anomalies:
            return input_tensor, target_tensor, input_anomalies_tensor
        return input_tensor, target_tensor

    def standardize(self, tensor: torch.Tensor, idx: int = None) -> torch.Tensor:
        """
        Standardizes the input tensor using the mean and standard deviation.
        """
        return (tensor - self.mean) / self.std

    def retrain_batch_idxs(self, batches_per_exp):
        num_batches = (len(self.indices)//self.config.test_batch_size)+1
        split_idxs = [batch_idx-1 for batch_idx in range(batches_per_exp, num_batches, batches_per_exp)]
        return set(split_idxs)
    

    def getbatch(self, batch_idxs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a batch of (input_window, target_window, anomaly_labels) pairs.
        """
        input_batch, target_batch = [], []
        for segment_idx, start_pos in batch_idxs:
            segment = self.data_segments[segment_idx]

            # Extract the full sample slice
            sample_slice = segment[start_pos : start_pos + self.sample_len]

            # Split into input and target windows
            input_window = sample_slice[:self.window_size]
            target_window = sample_slice[self.window_size:]
            
            # Convert to tensor
            input_tensor = torch.from_numpy(input_window)
            target_tensor = torch.from_numpy(target_window)

            # Standardize the data
            input_batch.append((input_tensor - self.mean) / self.std)
            target_batch.append((target_tensor - self.mean) / self.std)

        return torch.stack(input_batch, dim=0), torch.stack(target_batch, dim=0)

class ReplayBuffer(TelemanomDataset):
    """
    A simple replay buffer to store training data for continual learning.

    """
    def __init__(self,
                 data_segments: List[np.ndarray],
                 mean: np.ndarray,
                 std: np.ndarray,
                 config: ExperimentConfig,
                 anomalies: Optional[List[np.ndarray]] = None,
                 output_old_preds: bool = False,
                 old_preds: List[np.ndarray] = []
                 ):
        super(ReplayBuffer, self).__init__(
                data_segments,
                mean,
                std,
                config,
                anomalies
                )
        
        self.output_old_preds = output_old_preds
        self.old_preds = old_preds

    def add_processed_data(self, new_train: List[np.ndarray], indices: List[Tuple[int, int]], old_predictions: List[np.ndarray] = []):

        if len(indices)<1:
            logging.warning("no data to add to the replay buffer")

        indice_start = len(self.data_segments)
        self.data_segments += [arr.copy() for arr in new_train]
        self.indices += [(indice_start+i, j) for i, j in indices]
        self.anomalies += [np.zeros(seg.shape[0]) for seg in new_train]
        self.old_preds += [pred.copy() for pred in old_predictions]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a single standardized (input_window, target_window, anomaly_labels) pair.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        # Map the flat index to the correct segment and position
        segment_idx, start_pos = self.indices[idx]
        segment = self.data_segments[segment_idx]
        anomaly_segment = self.anomalies[segment_idx]

        # Extract the full sample slice
        sample_slice = segment[start_pos : start_pos + self.sample_len]
        anomaly_slice = anomaly_segment[start_pos : start_pos + self.sample_len]

        # Split into input and target windows
        input_window = sample_slice[:self.window_size]
        target_window = sample_slice[self.window_size:]
        input_anomalies = anomaly_slice[:self.window_size]
        
        # Convert to tensor
        input_tensor = torch.from_numpy(input_window)
        target_tensor = torch.from_numpy(target_window)
        input_anomalies_tensor = torch.from_numpy(input_anomalies).long()

        # Standardize the data
        input_tensor = (input_tensor - self.mean) / self.std
        target_tensor = (target_tensor - self.mean) / self.std

        if self.output_old_preds:
            old_preds = torch.from_numpy(self.old_preds[segment_idx][start_pos])
            old_preds = (old_preds - self.mean) / self.std
            if self.output_anomalies:
                return input_tensor, target_tensor, input_anomalies_tensor, old_preds
            return input_tensor, target_tensor, old_preds

        if self.output_anomalies:
            return input_tensor, target_tensor, input_anomalies_tensor
        return input_tensor, target_tensor