import numpy as np
import torch
from typing import List
from tqdm import tqdm
import gc
import os
import logging
import time
import random
from src.base_model import BaseModel
from src.experiment_config import ExperimentConfig
from src.dataset_manager import DatasetManager
from src.metrics.MetricsEvaluator import MetricsEvaluator
from .CL_methods.continual_learning_manager import CL_Manager
from .general_utils import initial_training

def run(model: BaseModel, config: ExperimentConfig, dataset_manager_list: List[DatasetManager]):
    """
    Run the model with the given training data manager and configuration.

    Args:
        model: The model to run.
        trainset_manager: The data manager containing training data.
        config: Configuration parameters for the run.
    """
    init_traintime_dict = {}

    if config.pre_trained_model_path is not None:
        logging.info(f"Loading pre-trained model from {config.pre_trained_model_path}")
        model.load_state_dict(torch.load(config.pre_trained_model_path, map_location=config.device))
        logging.info("Pre-trained model loaded successfully. Skipping initial training.")
        torch.save(model.state_dict(), config.results_path / "best_model.pth")
    else:
        start_time = time.time()
        initial_training(model, dataset_manager_list[0], config)
        init_traintime_dict = {'init_train_time': time.time() - start_time}
    
    # Ensuring reproducibility
    set_random_seeds(config)

    testing_model = model
    logging.info(f"Testing on {config.collection}/{config.dataset} test set...")

    try:
        anomaly_predictions, times_dict = testing_loop(testing_model, dataset_manager_list[0], config)
        if init_traintime_dict:
            times_dict.update(init_traintime_dict)
        logging.info("Testing completed. Calculating metrics...")
    except Exception as e:
        logging.error(f"An error occurred during testing. Skipping to the next test set.", exc_info=True)

    try:
        evaluator = MetricsEvaluator(config, y_hat=anomaly_predictions, execution_times=times_dict, metrics_df=dataset_manager_list[0].metrics_df) # if anomaly_predictions is None, it loads the csv
        evaluator.evaluate()
        logging.info("Metrics evaluation completed.")
    except Exception as e:
        logging.warning(f"Metrics evaluation failed with error: {e}\nSkipping metrics evaluation.")


def testing_loop(model: BaseModel, dataset_manager: DatasetManager, config: ExperimentConfig, compute_anomaly_predictions: bool = True):

    logging.info("Running the Testing Loop...")
    errors, targets, metadata = error_out_of_loop(model, dataset_manager, config)

    # saving files that are needed
    if config.save_errors:
        errors_file = config.results_path / "errors.csv"
        errors_save = np.concatenate((np.zeros((config.window_size, errors.shape[-1])), errors, np.zeros(((config.prediction_window_size - 1), errors.shape[-1])))) if config.model == "telemanom" else errors
        store_outputs(errors_save,
                      errors_file)

    # computing and saving anomaly_predictions, if needed
    anomaly_predictions = None
    if compute_anomaly_predictions:
        anomaly_predictions, thresholds = model.detect_anomalies(errors, targets)
        anomaly_predictions = np.concatenate((np.zeros((config.window_size, anomaly_predictions.shape[-1])), anomaly_predictions, np.zeros(((config.prediction_window_size - 1), anomaly_predictions.shape[-1]))), axis=0) if config.model == "telemanom" else anomaly_predictions

        if config.save_anomalies:
            anomaly_predictions_file = config.results_path / "anomaly_prediction.csv"
            logging.info(f"Shape of predicted anomalies after padding: {anomaly_predictions.shape}")
            store_outputs(anomaly_predictions,
                          anomaly_predictions_file)
        
        if config.save_thresholds:
            thresholds_file = config.results_path / "thresholds.csv"
            thresholds = np.concatenate((np.zeros((config.window_size, thresholds.shape[-1])), thresholds, np.zeros(((config.prediction_window_size - 1), thresholds.shape[-1]))), axis=0) if config.model == "telemanom" else thresholds
            logging.info(f"Shape of predicted anomalies after padding: {thresholds.shape}")
            store_outputs(thresholds,
                          thresholds_file)

    return anomaly_predictions, metadata


def error_out_of_loop(model: BaseModel, dataset_manager: DatasetManager, config: ExperimentConfig):
    """
    Run the Continual Learning loop with the given model and data.

    Args:
        init_model: The initial model to run.
        init_train_data: The initial training data.
        init_val_data: The initial validation data.
        config: Configuration parameters for the CL loop.
    """
    metadata = {} #store results
    # initializing replay buffer, if needed
    if config.continual_learning:
        logging.info("Running with Continual Learning")
        dataset_manager.init_replay_buffer(delete_train=False)  # ContinualLearningManager
        cl_manager = CL_Manager(
            dataset_manager=dataset_manager,
            model=model,
            config=config,
            )
    else: 
        logging.info("Running without Continual Learning")

    test_dataloader = dataset_manager.get_dataloader('test', config.test_batch_size, shuffle=False)

    progress_bar = tqdm(test_dataloader, desc=f"Testing")

    # Safety check
    if len(progress_bar) < 1:
            raise ValueError("Test set is empty. Please check your dataset.")

    aggregated_y_hat = []
    all_targets = []

    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        for batch_idx, batch in enumerate(progress_bar):
            inputs, targets, anomaly_inputs = batch
            inputs, targets, anomaly_inputs = inputs.to(config.device), targets.to(config.device), anomaly_inputs.to(config.device)

            # Forward pass
            y_hat = model(inputs).cpu()
            targets = targets.cpu()[:,0,:] if config.model == "telemanom" else targets.cpu()

            batch_y_hat = model.aggregate_predictions(y_hat.numpy())
            aggregated_y_hat.append(batch_y_hat)
            all_targets.append(targets.numpy())

            if config.continual_learning: # Everything under this `if` should be handled in ContinualLearningManager
                cl_manager.save_for_replay(inputs[:, 0, :].cpu().numpy(), anomaly_inputs[:, 0].cpu().numpy().copy())
                cl_manager.check_retraining(batch_idx=batch_idx)
    
    #free memory
    delete_object(test_dataloader)

    # aggregate and save predictions
    aggregated_y_hat = np.concatenate(aggregated_y_hat, axis=0)
    reconstruction = dataset_manager.unscale_data(aggregated_y_hat)
    logging.info(f"Shape of predictions: {reconstruction.shape}")
    delete_object(y_hat)
    if config.save_reconstructions:
        first_row, last_row = reconstruction[0], reconstruction[-1]
        reconstructions_file = config.results_path / "reconstruction.csv"
        reconstruction = np.concatenate(([first_row] * config.window_size, reconstruction, [last_row] * (config.prediction_window_size - 1))) if config.model == "telemanom" else reconstruction
        store_outputs(reconstruction,
                      reconstructions_file, 
                      config.target_channels)
    
    all_targets = np.concatenate(all_targets, axis=0)

    unprocessed_errors = np.abs(aggregated_y_hat - all_targets)
    metadata.update(model.calc_error_percentile_score(unprocessed_errors))
    if config.save_unprocessed_errors:
        unprocessed_unpadded_errors_file = config.results_path / "unprocessed_unpadded_errors.csv"
        store_outputs(unprocessed_errors, unprocessed_unpadded_errors_file, config.target_channels)

    errors = model.compute_errors(aggregated_y_hat, all_targets)
    logging.info(f"Shape of errors: {errors.shape}")
    delete_object(aggregated_y_hat)
    delete_object(reconstruction)

    if config.continual_learning:
        metadata['total_cl_time'] = cl_manager.total_time

    return errors, all_targets, metadata

def store_outputs(outputs, save_path, channel_ids=None):
    """
    Store the outputs of the model in a specified format.

    Args:
        outputs: The outputs to store.
        config: Configuration parameters for storing outputs.
    """
    if os.path.isfile(save_path):
        raise ValueError("Reconstruction file already exists. Please delete it before running again.")
    
    if channel_ids is None:
        np.savetxt(
            save_path,
            outputs,
            delimiter=",",
        )

    else:
        np.savetxt(
            save_path,
            outputs,
            delimiter=",",
            header=",".join(channel_ids),
            comments="",
        )
    return

def set_random_seeds(config):
    random.seed(config.seed_random)
    np.random.seed(config.seed_np)
    torch.manual_seed(config.seed_torch)
    torch.cuda.manual_seed_all(config.seed_torch)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def delete_object(this_object):
    del this_object
    gc.collect()