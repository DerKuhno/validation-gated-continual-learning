import numpy as np
import pandas as pd
from pathlib import Path
import os
from dataclasses import asdict
from sklearn.preprocessing import MinMaxScaler
import time
import logging
import copy

from .utils.encode_params import dump_params

METRICS_CSV = "metrics.csv"
HYPER_PARAMETERS = "hyper_params.json"
RESULTS_CSV = "results.csv"

class MetricsEvaluator():
    """
    stores the predictions, true labels, experiment times, hyperparameters and results path
    and provides an evaluate method to calculate the metrics.

    Args:
        metric_list: List of metrics to evaluate.
        y_hat: Predictions from the model.
        true_label_path: Path to the true labels.
        experiment_times: Dictionary of experiment times.
        hyperparams: Dictionary of hyperparameters.
        results_path: Path to store the results.
    
    Usage:
        evaluator = MetricsEvaluator(metric_list, y_hat, true_label_path, experiment_times, hyperparams, results_path)
        results = evaluator.evaluate()
    """
    def __init__(self,
                 config,
                 y_hat = None, # if y_hat is not used, then the predictions are loaded from the csv
                 execution_times: dict = {},
                 metrics_df: pd.DataFrame = None,
            ):

        self.results_dict = {}
        self.metrics = copy.deepcopy(config.metrics)

        self.execution_times = execution_times
        self.results_path = config.results_path
        self.channel_names = config.target_channels

        if isinstance(self.results_path, str):
            self.results_path = Path(self.results_path)

        if y_hat is not None:
            self.predictions = self.scale_scores(y_hat)
        elif os.path.exists(self.results_path / "anomaly_prediction.csv"):
            prediction_path = Path(self.results_path / "anomaly_prediction.csv")
            self.predictions = np.loadtxt(prediction_path, delimiter=",")
            self.predictions = self.scale_scores(self.predictions)
        else: raise ValueError("Please input either the predictions or a working csv path to MetricsEvaluator")
        if self.predictions.ndim == 1:
            self.predictions = np.expand_dims(self.predictions, -1)

        # saves config to json file
        print_config = asdict(config)
        allowed_types = (str, int, float, bool, type(None))
        for key in list(print_config.keys()):
            value = print_config[key]
            if isinstance(value, allowed_types):
                continue # keep it
            elif isinstance(value, list):
                if all(isinstance(item, allowed_types) for item in value):
                    continue  # keep it
            elif isinstance(value, dict):
                for dict_key, dict_item in value.items():
                    if isinstance(dict_item, allowed_types):
                        print_config[dict_key] = dict_item  # keep it
            del print_config[key]
        dump_params(print_config, self.results_path / HYPER_PARAMETERS)
        
        # creating the self.labels_df, self.prepared_labels_df, self.test_data_scores
        logging.info("Start loading the different dataframes from the CSV's...")
        self.labels_df: pd.DataFrame = pd.read_csv(
            config.labels_csv_path, parse_dates=["StartTime", "EndTime"]
        )
        self.labels_df["StartTime"] = self.labels_df["StartTime"].apply(
            lambda t: t.tz_localize(None)
        )
        self.labels_df["EndTime"] = self.labels_df["EndTime"].apply(
            lambda t: t.tz_localize(None)
        )

        if metrics_df is None:
            self.test_data_scores = pd.read_csv(
                config.test_dataset_path, usecols=["timestamp"], parse_dates=[0]
            )
            self.test_data_scores.rename(
                columns={"timestamp": "Timestamp"}, inplace=True
            )
        else:
            self.test_data_scores = metrics_df
        self.test_data_scores["Score"] = np.uint8(0)

        self.labels_df = self.labels_df[
            self.labels_df["StartTime"]
            >= self.test_data_scores["Timestamp"].min()
        ]
        self.labels_df = self.labels_df[
            self.labels_df["EndTime"]
            <= self.test_data_scores["Timestamp"].max()
        ]

        anomaly_types_df: pd.DataFrame = pd.read_csv(config.anomaly_types_path)
        columns_to_copy = anomaly_types_df.columns[-4:]
        for col in columns_to_copy:
            self.labels_df[col] = ""
        for index, row in anomaly_types_df.iterrows():
            self.labels_df.loc[
                self.labels_df["ID"] == row["ID"], columns_to_copy
            ] = row[columns_to_copy].values
        self.prepared_labels_df = self.labels_df[self.labels_df["Channel"].isin(self.channel_names)]
        
        channels_df = pd.read_csv(config.channels_path)
        self.subsystems_mapping = {
            s: [*v] for s, v in channels_df.groupby("Subsystem")["Channel"]
        }
        logging.info("Finished loading the different dataframes from the CSV's!")
        print("all dataframes loaded!")


    def evaluate(self) -> dict:

        results = {} # the final results dictionary
        results.update(self.execution_times)
        # backup results to disk
        pd.DataFrame([results]).to_csv(self.results_path / METRICS_CSV, index=False)
        

        errors = 0
        last_exception = None

        # Check if only global "is_anomaly" or per channel predictions are available
        only_global_scores = False
        if self.predictions.shape[1] == 1:
            only_global_scores = True
            print("Only global predictions available or self.prediction has just 1 dimension")

        # First calculate global metrics (loops through all metrics)
        for metric in self.metrics:
            start = time.time()
            if hasattr(metric, "_plot_store"):
                metric._plot_store = self.results_path

            try:
            
                self.test_data_scores["Score"] = self.predictions.max(axis=1).astype(
                    np.uint8
                )
                score = metric.score(
                    self.prepared_labels_df.drop(columns=["Channel"]), self.test_data_scores
                )
                

                if isinstance(score, dict):
                    for submetric, value in score.items():
                        print(f"{metric.name}_{submetric} calculated: {value}")
                        results[f"{metric.name}_{submetric}"] = value
                else:
                    print(f"{metric.name} calculated: {score}")
                    results[f"{metric.name}"] = score

            except Exception as e:
                print(f"Exception while computing metric {metric.name}: {e}")
                errors += 1
                if str(e):
                    last_exception = e
                continue
            print(f"{metric.name} took {time.time() - start:.4f} seconds")


        # write all results to disk (overwriting backup)
        pd.DataFrame([results]).to_csv(self.results_path / METRICS_CSV, index=False)

        self.results_dict = results

        # rethrow exception if no metric could be calculated
        if errors == len(self.metrics) and last_exception is not None:
            raise last_exception

        return results
    
    @staticmethod
    def scale_scores(y_scores: np.ndarray) -> np.ndarray:
        y_scores = np.asarray(y_scores, dtype=np.float32)

        if y_scores.ndim == 1:
            y_scores = np.expand_dims(y_scores, -1)

        for i in range(y_scores.shape[-1]):
            # mask NaNs and Infs
            mask = (
                np.isinf(y_scores[..., i])
                | np.isneginf(y_scores[..., i])
                | np.isnan(y_scores[..., i])
            )

            # scale all other scores to [0, 1]
            scores = y_scores[..., i][~mask]
            if scores.size != 0:
                if len(scores.shape) == 1:
                    scores = scores.reshape(-1, 1)
                y_scores[..., i][~mask] = MinMaxScaler().fit_transform(scores).ravel()

        return y_scores