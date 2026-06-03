import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np
import more_itertools as mit
import gc
from time import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d
import copy

# Get the absolute path to the parent folder (one level up from 'notebook/')
parent_dir = os.path.abspath("/root/The_Only_Work_Directory/1. pipelines_clfl")

# Add parent_dir to Python's module search path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.dataset_manager import DatasetManager
from src.metrics.MetricsEvaluator import MetricsEvaluator
from src.experiment_config import ExperimentConfig
from src.telemanom_model import TelemanomModel

DEFAULT_CHANNELS = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]

# Absolute path to the directory of ESA-ADB
#external_dir = '/root/The_Only_Work_Directory/ESA-ADB/TimeEval-algorithms/telemanom_esa'
#sys.path.append(external_dir)

config = ExperimentConfig(
    hyperparameter_search_split=True,
    z_start=1,
    dataset="84_months",
    val_batch_size=70,

    #save_reconstructions=False,
    #last_three_months=True,
    pre_trained_model_path = "/root/The_Only_Work_Directory/1. pipelines_clfl/results/3Months-Validation/best_model.pth",
    dynamic_thresholding = True,
    rolling_window_stride = 70,
    rolling_buffer = 200
)

#model = TelemanomModel(config).to(config.device)

#model.load_state_dict(torch.load(config.pre_trained_model_path, map_location=config.device))

processed_data_folder = Path(parent_dir) / "data/preprocessed"
dataset_manager = DatasetManager(
        config=config,
        include_test=True,  # True: loads test data
        client_idx=0,
    )

unprocessed_errors = np.loadtxt("/root/The_Only_Work_Directory/1. pipelines_clfl/results/UPPER_BOUND/unprocessed_unpadded_errors.csv", delimiter=",", skiprows=1)
unprocessed_errors = (unprocessed_errors / dataset_manager.test.std)
print(unprocessed_errors.shape)

def calculate_val_predictions(model, eval_dataloader, device, epoch, dataset_manager):
    high_validation_loss = 0
    
    val_preds = []
    all_targets = []
    model.eval()

    val_progress_bar = tqdm(eval_dataloader, desc=f"Validation predictions")

    start_time = time()
    with torch.no_grad():
        for val_batch in val_progress_bar:
            val_inputs, val_targets, _ = val_batch
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            val_preds.append(model(val_inputs).cpu() * dataset_manager.test.std + dataset_manager.test.mean)
            all_targets.append(val_targets.cpu()[:,0,:]  * dataset_manager.test.std + dataset_manager.test.mean)

    end_time = time()
    print(f"Experience epoch {epoch+1} validation completed in {end_time - start_time:.2f} seconds.")

    val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
    aggregated_val_preds = model.aggregate_predictions(val_preds)
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    return aggregated_val_preds, all_targets

def calculate_val_errors(model, aggregated_predictions, targets):
    return model.compute_errors(aggregated_predictions, targets)

def detect_anomalies_dynamic_threshold(model, errors, targets):
    return model.detect_anomalies(errors, targets)

def detect_anomalies_rolling_window(errors):
    window_mean = config.error_analysis_window_size
    window_std  = config.error_analysis_window_size
    frac_std = config.z_start
    n, m = errors.shape

    errs = errors.astype(float)

    # Rolling mean
    cs_mean = np.cumsum(errs, axis=0)
    cs_mean_pad = np.vstack([np.zeros((1, m)), cs_mean])
    sums_mean = cs_mean_pad[window_mean:] - cs_mean_pad[:-window_mean]
    mean = sums_mean / window_mean  # (n - window_mean + 1, m)

    # Rolling std (via E[X^2] - E[X]^2)
    cs_std  = np.cumsum(errs, axis=0)
    cs2_std = np.cumsum(errs**2, axis=0)
    cs_std_pad  = np.vstack([np.zeros((1, m)), cs_std])
    cs2_std_pad = np.vstack([np.zeros((1, m)), cs2_std])
    sums_std  = cs_std_pad[window_std:]  - cs_std_pad[:-window_std]
    sumsq_std = cs2_std_pad[window_std:] - cs2_std_pad[:-window_std]
    mean_for_std = sums_std / window_std
    var = sumsq_std / window_std - mean_for_std**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)  # (n - window_std + 1, m)

    # Align mean & std arrays to same shape before combining
    # Here we simply trim both to min length
    min_len = min(len(mean), len(std))
    mean = mean[-min_len:]
    std  = std[-min_len:]

    core = mean + frac_std * std

    # Prepend first row to match n rows
    head = np.repeat(core[:1, :], n - len(core), axis=0)
    thresholds = np.vstack([head, core])
    anoms = errors > thresholds

    # Prune singletons anywhere
    # prev = np.vstack([np.zeros((1, m), dtype=bool), anoms[:-1]])
    # next_ = np.vstack([anoms[1:], np.zeros((1, m), dtype=bool)])
    # singleton_mask = anoms & ~prev & ~next_
    # pruned = anoms & ~singleton_mask

    return anoms, thresholds

def detect_anomalies_straight_line(errors):
    std, mean = errors.std(axis=0), errors.mean(axis=0)
    thresholds = mean + config.z_start * std
    expanded_thresholds = np.expand_dims(thresholds, axis=0)
    return errors > thresholds, np.repeat(expanded_thresholds, repeats=len(errors), axis=0)

def calculate_thresholds(errors: np.ndarray):
    """
    Calculate per-channel anomaly thresholds using a sliding window.
    
    Parameters
    ----------
    errors : np.ndarray
        Array of shape (N, 3) with error values.
    window_size : int
        Size of the sliding window.
    
    Returns
    -------
    anomalies : np.ndarray
        Boolean array of shape (N, 3) where True means anomaly.
    thresholds : np.ndarray
        Float array of shape (N, 3) with calculated thresholds.
    """
    N, C = errors.shape
    window_size = config.rolling_window_size
    thresholds = np.full((N, C), np.nan)
    anomalies = np.zeros((N, C), dtype=bool)

    for start_idx in tqdm(range(0, N, config.rolling_window_stride), desc=f"Threshold calculation"):
        # Determine the window range
        window_start = max(0, start_idx - window_size + 1)
        window_end = min(start_idx + config.rolling_window_stride, N)
        window = errors[window_start:window_end, :]  # shape: (win_len, C)

        # Sort and get lowest 99% per channel
        sorted_vals = np.sort(window, axis=0)
        cutoff_index = int(np.floor(config.rolling_calc_percentile * sorted_vals.shape[0]))
        lowest_99 = sorted_vals[:cutoff_index, :] if cutoff_index > 0 else sorted_vals

        # Mean and std per channel
        mean_val = np.mean(lowest_99, axis=0)
        std_val = np.std(lowest_99, axis=0, ddof=0)
        threshold = config.mean_mul * mean_val +  config.rolling_threshold_std * std_val

        # Apply threshold to the whole batch
        batch_end = min(start_idx + config.rolling_window_stride, N)
        thresholds[start_idx:batch_end, :] = threshold
    
    thresholds[:2*config.window_size] = thresholds[2*config.window_size]

    anomalies = errors > thresholds

    # This padds 50 values to the front and back of an anomaly
    anomalies = maximum_filter1d(anomalies.astype(int), size=config.rolling_buffer+1, axis=0, mode="constant") > 0

    return anomalies, thresholds

def pad_array(arr):
    return np.concatenate((np.zeros((config.window_size, arr.shape[-1])), arr, np.zeros(((config.prediction_window_size - 1), arr.shape[-1]))))

#val_df = dataset_manager.val_df
# validation_dataloader = dataset_manager.get_dataloader(
#                                                         dataset='test', 
#                                                         batch_size=config.val_batch_size, 
#                                                         shuffle=False,
#                                                         )
# aggregated_val_preds, all_targets = calculate_val_predictions(
#                                                                 model,
#                                                                 validation_dataloader,
#                                                                 config.device,
#                                                                 0,  # epoch
#                                                                 dataset_manager,
#                                                             )

unprocessed_errors = pad_array(unprocessed_errors)

# del validation_dataloader, aggregated_val_preds, all_targets

# Hyperparameter search
test_idx = 0
val_df = dataset_manager.metrics_df.copy()
del dataset_manager
config.results_path = Path("/root/The_Only_Work_Directory/1. pipelines_clfl/results/threshold_finding_upper_bound")

# retrieving best dict
best_0_5_score = 0
best_results = {}
best_params = {}
best_idx = 0

for perc in np.arange(0.001, 0.03, 0.006):
    smoothing_window_size = int(
            config.batch_size
            * config.smoothing_span
            * perc
        )
    light_processed_errors = pd.DataFrame(unprocessed_errors).ewm(span=smoothing_window_size).mean().values
    for rolling_window_size in [5000, 10000, 50000]:
        config.rolling_window_size = rolling_window_size
        # if test_idx > 142:
        #         break
        for rolling_threshold_std in np.arange(2.5, 9.5, 1):
            config.rolling_threshold_std = rolling_threshold_std
            # if test_idx > 142:
            #     break
            for mean_mul in np.arange(1, 2.3, 0.3):
                # if test_idx > 142:
                #     break
                config.mean_mul = mean_mul
                parameter_dict = {"perc": perc, "rolling_threshold_std": rolling_threshold_std, "mean_mul": mean_mul, "rolling_window_size": rolling_window_size}
                print(f"working on idx: {test_idx}")
                percentile_anomalies, percentile_thresholds = calculate_thresholds(light_processed_errors)
                evaluator = MetricsEvaluator(
                    config,
                    percentile_anomalies,
                    parameter_dict,
                    val_df,
                    test_idx,
                )
                evaluator.evaluate()

                if evaluator.results_dict["Rare Event_Anomaly_EW_F_0.50"] > best_0_5_score:
                    best_0_5_score = evaluator.results_dict["Rare Event_Anomaly_EW_F_0.50"]
                    best_results = evaluator.results_dict
                    best_params = parameter_dict
                    best_idx = test_idx
                print("best_0_5_score", best_0_5_score)
                print("best_params", best_params)
                print("best_idx", best_idx)
                test_idx +=1


print("best_0_5_score", best_0_5_score)
print("best_params", best_params)
print("best_idx", best_idx)