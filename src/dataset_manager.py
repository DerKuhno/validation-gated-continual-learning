import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm
from src.experiment_config import ExperimentConfig
from src.dataset import TelemanomDataset, ReplayBuffer
from src.esaadb_dataset import ESAADBDataset
import gc

ROOT = Path(__file__).parent.parent

class DatasetManager:
    """
    Manages the creation of training, validation, and test datasets and dataloaders.
    """
    def __init__(self,
                 config: ExperimentConfig = ExperimentConfig(),
                 include_test: bool = True,
                 nrows: Optional[int] = None
                 ):
        """
        Initializes the dataset manager by creating the datasets.

        Args:
            data_folder_path (str): Path to the preprocessed data folder.
            config (ExperimentConfig): Experiment configuration object.
            include_test (bool): Whether to include the test dataset.
            nrows (Optional[int]): Number of rows to read from the dataset. If None, reads all rows. Only for debugging purposes.
        """
        self.config = config

        if config.collection in ["ESA-Mission1", "ESA-Mission2"]:
            data_folder_path = os.path.join(ROOT, "data", "preprocessed")
            loader_func = ESAADBDataset.from_config
            loader_kwargs = {'data_folder_path': data_folder_path, 'nrows': nrows}
            loader_kwargs['include_test'] = include_test
        else:
            raise ValueError(f"Unknown collection: {config.collection}")

        all_data = TelemanomDataset.create_or_load(
            config=config,
            loader_func=loader_func,
            **loader_kwargs
        )

        def wrap_dataset(data):
            if data is None:
                return None
            elif self.config.model == 'telemanom':
                return data

        self.train = wrap_dataset(all_data[0])
        self.val = wrap_dataset(all_data[1])
        self.test = wrap_dataset(all_data[2])
        self.metrics_df = all_data[3]

    def get_dataloader(self, dataset: str, batch_size: int, shuffle: bool, start_idx: int = 0) -> Optional[DataLoader]:
        """
        Returns a DataLoader for the specified dataset.

        Args:
            dataset (str): The dataset to get a DataLoader for. Must be one of 'train', 'val', or 'test'.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            Optional[DataLoader]: A DataLoader for the specified dataset, or None if the dataset does not exist.
        """
        effective_sat_idx: Optional[int] = None
        if dataset == 'train':
            data = self.train
        elif dataset == 'val':
            data = self.val
        elif dataset == 'test':
            data = self.test
        elif dataset == 'tmp':
            data = self.tmp_dataset
        elif dataset == 'replay':
            data = self.replay_buffer
        else:
            raise ValueError("dataset must be one of 'train', 'val', 'test', 'tmp', or 'replay'.")

        if data is None:
            return None

        if start_idx > 0:
            data = Subset(data, list(range(start_idx*self.config.test_batch_size, len(data))))

        loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
        return loader

    def unscale_data(self, data: np.ndarray, test_set_idx: Optional[int] = None) -> np.ndarray:
        """
        Unscales the data using the mean and std of the test set.

        Args:
            data (np.ndarray): The data to unscale. Assumed to be on CPU.
            test_set_idx (Optional[int]): The index of the test set to use for unscaling.

        Returns:
            np.ndarray: The unscaled data.
        """
        if isinstance(self.test, list):
            if test_set_idx is None:
                logging.warning("test_set_idx is None but there are multiple test sets. Using index 0.")
                test_set_idx = 0
            std = self.test[test_set_idx].std
            mean = self.test[test_set_idx].mean
        else:
            std = self.test.std
            mean = self.test.mean

        std = std.numpy()
        mean = mean.numpy()
        return data * std + mean
    
    # def merge_datasets(self, datasets: List[TelemanomDataset]) -> TelemanomDataset:
    #     pass
    #     """
    #     Merges multiple TelemanomDataset instances into a single dataset.

    #     Args:
    #         datasets (List[TelemanomDataset]): List of TelemanomDataset instances to merge.

    #     Returns:
    #         TelemanomDataset: A new TelemanomDataset instance containing all data.
    #     """
    #     if not datasets:
    #         return TelemanomDataset([], np.array([]), np.array([]), self.config)

    #     merged_data_segments = []
    #     for dataset in datasets:
    #         merged_data_segments.extend(dataset.data_segments)

    #     # Use the mean/std from the first dataset
    #     mean = datasets[0].mean.numpy()
    #     std = datasets[0].std.numpy()

    #     return TelemanomDataset(merged_data_segments, mean, std, self.config)


    def extend_dataset(self, new_data: List[np.ndarray], anomaly_flag: List[np.ndarray], validationsplit: float):
        """
        Extend the dataset with new data segments. Used in the Replay Buffer

        Args:
            new_data (List[np.ndarray]): New data in batches.
            outputs (List[np.ndarray]): the outputs of the model
            anomaly_flag (List[np.ndarray]): tells where anomalies are in batches.
            validationsplit (float): Percentage of validation data
        """
        ##  checks
        if not isinstance(new_data, list) or not all(isinstance(seg, np.ndarray) for seg in new_data):
            raise ValueError("new_data must be a list of numpy arrays.")
        if not isinstance(anomaly_flag, list) or not all(isinstance(seg, np.ndarray) for seg in anomaly_flag):
            raise ValueError("anomaly_flag must be a list of numpy arrays.")
        if len(new_data)==0:
            logging.warning("Not enough data to add to the generator. Please provide at least 1 fragment.")
            return
        

        # preparing data into segments
        std = self.test.std.numpy()
        mean = self.test.mean.numpy()
        new_data = (np.concatenate(new_data, axis=0) * std) + mean # rescaling the normalized values
        anomaly_flag = np.concatenate(anomaly_flag, axis=0)
        assert len(new_data) == len(anomaly_flag), "data and anomalies must have the same length"
        assert anomaly_flag.ndim == 1, "wrong dimension for anomalies"
        val_idx = int(np.floor(len(new_data)*(1-validationsplit)))

        # Find indices where anomalies == 1
        anomaly_indices = np.flatnonzero(anomaly_flag)

        # preparing the input data 
        segments = []
        start_idx = 0

        for idx in anomaly_indices:
            if idx > start_idx:
                # Save the clean segment before the anomaly
                segments.append(new_data[start_idx:idx])
            start_idx = idx + 1  # skip the anomaly

        # Add the final clean segment (if any)
        if start_idx < len(new_data):
            segments.append(new_data[start_idx:])
        new_data = segments

        # adding to training and validation
        last_num_train_segments = len(self.tmp_dataset.data_segments)
        last_num_val_segments = len(self.val.data_segments)
        logging.info(f"starting with num datasegments in tmp: {last_num_train_segments} and validation segments: {last_num_val_segments}")
        if len(new_data) > 0:
            logging.info(f"shape of new_data:  {len(new_data)}, {new_data[0].shape}")
        else:
            logging.info("new_data is empty.")

        start_idx = 0
        for segment in new_data:
            end_idx = start_idx + len(segment)
            if end_idx < val_idx:
                self.tmp_dataset.data_segments.append(segment)
            elif start_idx > val_idx:
                self.val.data_segments.append(segment)
            else:
                if val_idx-start_idx>0:
                    self.tmp_dataset.data_segments.append(segment[:val_idx-start_idx])
                if end_idx-val_idx>0:
                    self.val.data_segments.append(segment[val_idx-start_idx:])
            start_idx = end_idx

        # adding new train indices
        for seg_idx in range(last_num_train_segments, len(self.tmp_dataset.data_segments)):
            seg_len = len(self.tmp_dataset.data_segments[seg_idx])
            if seg_len >= self.tmp_dataset.sample_len:
                num_windows = seg_len - self.tmp_dataset.sample_len + 1
                # Store as (segment_index, start_position_in_segment)
                self.tmp_dataset.indices.extend([(seg_idx, i) for i in range(num_windows)])

        # adding new val indices
        for seg_idx in range(last_num_val_segments, len(self.val.data_segments)):
            seg_len = len(self.val.data_segments[seg_idx])
            if seg_len >= self.val.sample_len:
                num_windows = seg_len - self.val.sample_len + 1
                # Store as (segment_index, start_position_in_segment)
                self.val.indices.extend([(seg_idx, i) for i in range(num_windows)])

        logging.info(f"shapes of tmp: {len(self.tmp_dataset.data_segments)}, {self.tmp_dataset.data_segments[0].shape}")
        logging.info(f"shapes of replay: {len(self.replay_buffer.data_segments)}, {self.replay_buffer.data_segments[0].shape}")


        self.tmp_dataset.anomalies += [np.zeros(self.tmp_dataset.data_segments[seg].shape[0]) for seg in range(len(self.tmp_dataset.data_segments) - last_num_train_segments)]
        self.val.anomalies += [np.zeros(self.val.data_segments[seg].shape[0]) for seg in range(len(self.val.data_segments) - last_num_val_segments)]

        logging.info(f"Check: Mean: {self.tmp_dataset.mean}, Check: Std: {self.tmp_dataset.std}")
        logging.info(
                    "Check if the indices are valid {} == {} if {} >= 260?".format(
                        len(self.tmp_dataset.data_segments) - 1, self.tmp_dataset.indices[-1][0], len(self.tmp_dataset.data_segments[-1])
                    )
                )
        logging.info(f"Check last indice: {self.tmp_dataset.indices[-1]}")
        return

    
    def init_replay_buffer(self, delete_train: bool = True, test_set_idx: Optional[int] = None):
        if test_set_idx is not None:
            mean = self.test[test_set_idx].mean.numpy().copy()
            std = self.test[test_set_idx].std.numpy().copy()
        else:
            mean = self.test.mean.numpy().copy()
            std = self.test.std.numpy().copy()

        # creates a replay_buffer (TelemanomDataset) and a Tmp_Dataset (TelemanomDataset)
        logging.info("Initializing the Replay Buffer")
        self.replay_buffer = ReplayBuffer(
            data_segments = self.train.data_segments.copy(),
            mean = mean,
            std = std,
            config = self.config
        )
        logging.info("Replay Buffer initialized")

        logging.info("Initializing the temporary dataset")
        self.tmp_dataset = ReplayBuffer(
            data_segments = [],
            mean = mean,
            std = std,
            config = self.config
        )
        logging.info("Temporary dataset initialized")

        #delete the train data to free memory
        if delete_train:
            self.train.data_segments = []
            self.train.anomalies = []
            self.train.indices = []
            gc.collect()
    
    def reset_tmp(self):
        self.tmp_dataset.data_segments = []
        self.tmp_dataset.indices = []
        self.tmp_dataset.anomalies = []
        self.tmp_dataset.old_preds = []
        self.tmp_dataset.output_old_preds = False

    def init_old_predictions(self, model):
        """
        This function tells the replay buffer to output old prediction and recomputes these.
        Must be called after initializing the replay buffer
        """
        device = self.config.device
        replay_dataloader = self.get_dataloader(dataset="replay", 
                                                batch_size=self.config.val_batch_size,
                                                shuffle=False)
    
        model.eval()
        progress_bar = tqdm(replay_dataloader, desc=f"Initial recompute of outputs")

        all_predictions = []
        with torch.no_grad():
            for idx, (mb_x, mb_y) in enumerate(progress_bar):
                mb_x = mb_x.to(device)
                all_predictions.append(model(mb_x).cpu())

        all_predictions = (torch.cat(all_predictions, dim=0) * self.replay_buffer.std + self.replay_buffer.mean).to(dtype=torch.float32).cpu().numpy().astype(np.float32)

        self.replay_buffer.old_preds = [np.zeros((len(seg), self.config.prediction_window_size, len(self.config.target_channels)), dtype=np.float32) for seg in self.replay_buffer.data_segments]

        for pred, (seg_idx, inner_idx) in zip(all_predictions, self.replay_buffer.indices):
            self.replay_buffer.old_preds[seg_idx][inner_idx] = pred

        model.train()
        self.replay_buffer.output_old_preds = True    
        return

    def recalculate_exp_predictions(self, model):
        """
        This function recomputes the predictions of the .
        Must be called after initializing the replay buffer
        """
        device = self.config.device
        tmp_dataloader = self.get_dataloader(dataset="tmp", 
                                                batch_size=self.config.val_batch_size,
                                                shuffle=False)
        
        model.eval()
        progress_bar = tqdm(tmp_dataloader, desc=f"Experience recompute of outputs")

        all_predictions = []
        with torch.no_grad():
            for idx, (mb_x, mb_y) in enumerate(progress_bar):
                mb_x = mb_x.to(device)
                all_predictions.append(model(mb_x).cpu())

        all_predictions = (torch.cat(all_predictions, dim=0) * self.tmp_dataset.std + self.tmp_dataset.mean).cpu().numpy()

        self.tmp_dataset.old_preds = [np.zeros((len(seg), self.config.prediction_window_size, len(self.config.target_channels)), dtype=np.float32) for seg in self.tmp_dataset.data_segments]

        for pred, (seg_idx, inner_idx) in zip(all_predictions, self.tmp_dataset.indices):
            self.tmp_dataset.old_preds[seg_idx][inner_idx] = pred

        model.train()
        self.tmp_dataset.output_old_preds = True  
        return
    
    def get_data_stream(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Converts a batch of inputs or outputs to the original stream format.

        Args:
            batch_data (np.ndarray): Batch of inputs or outputs from the model.

        Returns:
            np.ndarray: The data stream in original format.
        """
        if self.config.model == "telemanom":
            return batch_data
        else:
            raise ValueError(f"Unknown model type: {self.config.model}")
