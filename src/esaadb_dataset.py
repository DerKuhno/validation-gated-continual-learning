# FILE: dataset.py
import pandas as pd
from time import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging
from src.dataset import TelemanomDataset, get_split_timestamp
from src.experiment_config import ExperimentConfig
import gc

PREPROCESSED_DATA_FOLDER = Path(__file__).parent.parent / "data/preprocessed"

class ESAADBDataset(TelemanomDataset):
    """
    A PyTorch Dataset class to load and prepare the ESA-ADB time series data
    specifically for the Telemanom-ESA model.

    This class handles:
    - Loading data from an index file ('datasets.csv').
    - Identifying and extracting anomaly-free segments for training.
    - Splitting data into training and validation sets.
    - Calculating normalization statistics (mean/std) from the training set only.
    - Creating sliding window samples of (input_window, target_window).

    The primary way to use this class is via the `from_config` classmethod,
    which performs all necessary preparation and returns configured Dataset
    instances for training and validation.
    """
    INDEX_FILENAME = "datasets.csv"

    @classmethod
    def from_config(cls,
                   data_folder_path: str = PREPROCESSED_DATA_FOLDER,
                   config: ExperimentConfig = ExperimentConfig(),
                   include_test: bool = False,
                   nrows: Optional[int] = None,  # For debugging purposes
                   ) -> Union[Tuple, List]:
        """
        Factory method to create train/validation and optionally test dataset instances.

        Args:
            data_folder_path (str): Path to the root data folder.
            config (ExperimentConfig): The configuration object for the experiment.
            include_test (bool): If True, loads test data in addition to train/val data.

        Returns:
            If include_test is False:
                A tuple of (train_dataset, validation_dataset).
            If include_test is True:
                A tuple of (train_dataset, validation_dataset, test_dataset).
        """
        collection = config.collection
        dataset = config.dataset
        validation_split_ratio = config.validation_split_ratio
        input_channels = config.input_channels
        target_channels = config.target_channels

        root_path = Path(data_folder_path)
        index_path = root_path / cls.INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at '{index_path}'.")

        index_df = pd.read_csv(index_path).set_index(["collection_name", "dataset_name"])
        record = index_df.loc[(collection, dataset)]
        del index_df
        gc.collect()

        # --- Load Train Data ---
        train_relative_path = record["train_path"]
        if pd.isna(train_relative_path):
            raise ValueError(f"No train path for ('{collection}', '{dataset}').")
        train_full_path = root_path / train_relative_path

        # Common data preparation logic
        def prepare_data(full_path, load_full_data=True):

            all_columns = pd.read_csv(full_path, nrows=0).columns.tolist()
            
            anomaly_columns = [c for c in all_columns if c.startswith("is_anomaly")]
            
            data_columns = [c for c in all_columns[1:] if c not in anomaly_columns]

            if not load_full_data:
                return None, data_columns, None
            
            dtypes = {col: np.float32 for col in data_columns}
            
            dtypes.update({col: np.uint8 for col in anomaly_columns})
            
            df = pd.read_csv(full_path, index_col=0, dtype=dtypes, low_memory=True, chunksize=1_000_000, nrows=nrows)
            
            df = pd.concat(df)

            df.index = pd.to_datetime(df.index)

            
            final_anomaly_cols_in_df = [c for c in df.columns if c.startswith("is_anomaly")]
            
            return df, data_columns, final_anomaly_cols_in_df

        # Initial channel selection based on train data
        _, initial_data_columns, _ = prepare_data(train_full_path, load_full_data=False)
        if input_channels is None or len(set(input_channels).intersection(initial_data_columns)) == 0:
            input_channels = initial_data_columns
            logging.info(f"Input channels not given or not in data, selecting all channels and telecomands")
        else:
            input_channels = [c for c in input_channels if c in initial_data_columns]

        if target_channels is None or len(set(target_channels).intersection(initial_data_columns)) == 0:
            target_channels = [c for c in initial_data_columns if c.startswith("channel")]
            logging.info(f"Target channels not given or not in data, selecting all channels (not telecomands)")
        else:
            target_channels = [c for c in target_channels if c in initial_data_columns]
        
        all_used_channels = sorted(list(set(input_channels + target_channels)))

        # --- Process Train/Val Data ---
        start_time = time()
        df, _, final_anomaly_cols_in_df = prepare_data(train_full_path)
        print(f"Train-val data loaded after {time() - start_time:.2f} seconds.")
        logging.info(f"Train-val data loaded after {time() - start_time:.2f} seconds.")
        df = df.loc[:, all_used_channels + [c for c in final_anomaly_cols_in_df if c in df.columns]]

        # 1. Define target anomaly column
        target_anomaly_column = "is_anomaly_combined"
        df[target_anomaly_column] = 0
        
        target_anomaly_cols_to_check = [f"is_anomaly_{ch}" for ch in target_channels if f"is_anomaly_{ch}" in df.columns]

        if target_anomaly_cols_to_check:
            for col in target_anomaly_cols_to_check:
                df[target_anomaly_column] |= (df[col] > 0).astype(int)
        else:
            logging.warning("No anomaly columns found for target channels. Assuming all data is clean for segmentation.")

        df.drop(columns=[c for c in final_anomaly_cols_in_df if c in df.columns], inplace=True)

        # Continual Learning should start 16 Weeks before the testset
        if config.last_16_weeks:
            train_start = df.index[-(config.batches_per_exp * config.test_batch_size)] # 322560 datapoints = 16 weeks if no missing data
            target_date = df.index.max() - pd.DateOffset(weeks=16) + pd.DateOffset(seconds=30)
            print(f"Continual Learning will train on data after: {train_start} / target data check: {target_date}")
            logging.info(f"Continual Learning will train on data after: {train_start} / target data check: {target_date}")
            df = df[df.index >= train_start]
            print(f"and end with initial train data until: {df.index.max()}")
            logging.info(f"and end with initial train data until: {df.index.max()}")

        if config.hyperparameter_search_split:
            train_val_end = df.index[config.batches_per_exp * config.test_batch_size]
            target_date = df.index.min() + pd.DateOffset(weeks=16) - pd.DateOffset(seconds=30)
            print(f"Continual Learning will train on data after: {df.index.min()}")
            logging.info(f"Continual Learning will train on data after: {df.index.min()}")
            test_df_full = df[df.index >= train_val_end].copy()
            df = df[df.index < train_val_end]
            print(f"and end with initial train data until: {df.index.max()} / target data check: {target_date}")
            logging.info(f"and end with initial train data until: {df.index.max()} / target data check: {target_date}")
            print(f"It will test hyperparameters with data from: {test_df_full.index.min()}")
            logging.info(f"It will test hyperparameters with data from: {test_df_full.index.min()}")
            print(f"and end the test until: {test_df_full.index.max()}")
            logging.info(f"and end the test until: {test_df_full.index.max()}")


        client_dfs = [df]

        client_datasets = []
        for client_df in client_dfs:
            # 3a. Find all anomaly-free segments
            label_groups = client_df.groupby((client_df[target_anomaly_column].shift() != client_df[target_anomaly_column]).cumsum())
            
            clean_segments_df = [
                group.drop(columns=[target_anomaly_column])
                for _, group in label_groups
                if group[target_anomaly_column].iloc[0] == 0
            ]
            
            if not clean_segments_df:
                logging.warning(f"No anomaly-free segments found. Skipping.")
                client_datasets.append([None, None, None, None])
                continue

            # 3b. Split segments into train and validation
            validation_date_split = get_split_timestamp(client_df.index, validation_split_ratio)
            split_timestamp = pd.to_datetime(validation_date_split)
            train_segs = []
            val_segs = []

            for segment_df in clean_segments_df:

                start_date, end_date = segment_df.index[0], segment_df.index[-1]

                if end_date < split_timestamp:
                    train_segs.append(segment_df[all_used_channels].values.astype(np.float32))
                elif start_date > split_timestamp:
                    val_segs.append(segment_df[all_used_channels].values.astype(np.float32))
                else:
                    train_part = segment_df.loc[segment_df.index < split_timestamp]
                    val_part = segment_df.loc[segment_df.index >= split_timestamp]
                    if not train_part.empty:
                        train_segs.append(train_part[all_used_channels].values.astype(np.float32))
                    if not val_part.empty:
                        val_segs.append(val_part[all_used_channels].values.astype(np.float32))

            if not train_segs:
                logging.warning("Not enough anomaly-free segments for a training set.")
                client_datasets.append([None, None, None, None])
                continue
            if not val_segs:
                logging.warning("Not enough anomaly-free segments for a validation set after split.")

            # 3c. Calculate mean/std from training segments ONLY
            all_train_data = np.concatenate(train_segs, axis=0)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            std[std == 0] = 1

            # 3d. Create dataset instances
            train_dataset = cls(train_segs, mean, std, config)
            val_dataset = cls(val_segs, mean, std, config) if val_segs else None
            
            client_datasets.append([train_dataset, val_dataset, None, None])

        del client_dfs, all_train_data, clean_segments_df, train_part, val_part
        gc.collect()

        # --- Load Test Data ---
        logging.info(f"Loading test data")
        if include_test:
            test_relative_path = record["test_path"]
            if pd.isna(test_relative_path):
                logging.warning(f"No test path for ('{collection}', '{dataset}'). Test sets will be None.")
                test_dfs = [None]
                anomaly_dfs = [None]
            else:
                if not config.hyperparameter_search_split:
                    test_full_path = root_path / test_relative_path
                    start_time = time()
                    test_df_full, _, final_anomaly_cols_in_df = prepare_data(test_full_path)
                    print(f"Test data loaded after {time() - start_time:.2f} seconds.")
                    logging.info(f"Test data loaded after {time() - start_time:.2f} seconds.")

                    # Create combined anomaly column for test set
                    target_anomaly_column = "is_anomaly_combined"
                    test_df_full[target_anomaly_column] = 0
                    target_anomaly_cols_to_check = [f"is_anomaly_{ch}" for ch in target_channels if f"is_anomaly_{ch}" in test_df_full.columns]
                    if target_anomaly_cols_to_check:
                        for col in target_anomaly_cols_to_check:
                            test_df_full[target_anomaly_column] |= (test_df_full[col] > 0).astype(int)
                    
                    test_anomaly_data = test_df_full[target_anomaly_column].values
                    
                    # Drop all anomaly-related columns before creating the dataset
                    cols_to_drop = [target_anomaly_column] + [c for c in final_anomaly_cols_in_df if c in test_df_full.columns]
                    test_df_full.drop(columns=cols_to_drop, inplace=True, errors='ignore')
                else:
                    test_df_full = test_df_full
                    test_anomaly_data = test_df_full[target_anomaly_column].values
                    test_df_full.drop(columns=[target_anomaly_column], inplace=True, errors='ignore')
                    logging.info(f"Preceeding with the hyperparameter search from {test_df_full.index.min()} until {test_df_full.index.max()}")
                    print(f"Preceeding with the hyperparameter search from {test_df_full.index.min()} until {test_df_full.index.max()}")

                test_df_full = test_df_full[all_used_channels]
                test_dfs = [test_df_full]
                anomaly_dfs = [test_anomaly_data]

            for i, test_df in enumerate(test_dfs):
                print(f"Continual Learning will test on data after: {test_df.index.min()}, until: {test_df.index.max()}")
                if test_df is None or not client_datasets or client_datasets[i][0] is None:
                    client_datasets[i][2] = None
                    continue

                metrics_df = pd.DataFrame({"Timestamp": test_df.index.copy()})
                client_datasets[i][3] = metrics_df

                test_data_np = test_df.values.astype(np.float32)
                test_anomalies_np = anomaly_dfs[i]
                test_dates = test_df.index.to_list()
                
                # Use mean/std from the corresponding training set
                mean = client_datasets[i][0].mean.numpy()
                std = client_datasets[i][0].std.numpy()

                test_dataset = cls([test_data_np], mean, std, config, anomalies=[test_anomalies_np], output_anomalies=True, dates=test_dates)
                client_datasets[i][2] = test_dataset

        # Finalize return structure
        final_return = [tuple(cd) for cd in client_datasets]
        return final_return[0]