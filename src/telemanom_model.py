import torch
import torch.nn as nn
import numpy as np
from src.experiment_config import ExperimentConfig
from src.detector import Detector
from src.base_model import BaseModel

class TelemanomModel(BaseModel):
    """
    A PyTorch implementation of the Telemanom-ESA LSTM-based model.

    The model consists of two LSTM layers followed by a dense layer to predict
    a future window of time series values.
    """
    def __init__(self,
                 config: ExperimentConfig,
                 ):
        super(TelemanomModel, self).__init__()
        self.config = config
        self.num_target_channels = len(config.target_channels)
        self.num_input_channels = len(config.input_channels)
        self.detector = Detector(config)

        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=self.num_input_channels,
            hidden_size=config.lstm_layers[0],
            batch_first=True,  # Important: input format is (batch, seq, feature)
            num_layers=1
        )
        self.dropout1 = nn.Dropout(p=config.dropout)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=config.lstm_layers[0],
            hidden_size=config.lstm_layers[1],
            batch_first=True,
            num_layers=1
        )
        self.dropout2 = nn.Dropout(p=config.dropout)

        # Output layer
        self.dense = nn.Linear(
            in_features=config.lstm_layers[1],
            out_features=config.prediction_window_size * self.num_target_channels
        )

        if self.config.loss_function == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
        if self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, num_input_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, prediction_window_size, num_target_channels).
        """
        # Pass through first LSTM layer
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # We only need the full sequence output here
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)

        # Pass through second LSTM layer
        # For the second LSTM, we only care about the output of the final time step
        # to make the prediction.
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # last_hidden_state shape: (batch_size, hidden_size)
        lstm_out, _ = self.lstm2(lstm_out)
        last_hidden_state = lstm_out[:, -1, :]
        last_hidden_state = self.dropout2(last_hidden_state)

        # Pass through the fully connected layer
        dense_out = self.dense(last_hidden_state)

        # Reshape the output to (batch_size, prediction_window_size, num_target_channels)
        output = dense_out.view(-1, self.config.prediction_window_size, self.num_target_channels)

        return output

    def aggregate_predictions(self, y_hat, method='first'):
        if method == "ewma":
            ewma_weights = np.array([1 / np.exp(-x) for x in range(self.config.prediction_window_size, 0, -1)])

        agg_y_hat = np.zeros_like(y_hat[:, 0, :])

        for t in range(len(y_hat)):

            start_idx = t - self.config.prediction_window_size
            start_idx = start_idx if start_idx >= 0 else 0

            y_hat_channels = np.array([])
            for ch in range(y_hat.shape[-1]):
                # predictions pertaining to a specific timestep lie along diagonal
                y_hat_t = np.flipud(y_hat[start_idx:t+1, :, ch]).diagonal()

                if method == 'first':
                    y_hat_channels = np.append(y_hat_channels, [y_hat_t[0]])
                elif method == 'mean':
                    y_hat_channels = np.append(y_hat_channels, np.mean(y_hat_t))
                elif method == "ewma":
                    weights = ewma_weights[:len(y_hat_t)]
                    weights /= np.sum(weights)
                    y_hat_channels = np.append(y_hat_channels, np.sum(y_hat_t * weights))

            agg_y_hat[t] = y_hat_channels

        return agg_y_hat

    def compute_errors(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute errors between predicted and actual values.

        Args:
            y_hat (np.ndarray): Predicted values.
            y_true (np.ndarray): Actual values.

        Returns:
            np.ndarray: Errors for each channel.
        """
        return self.detector.compute_errors(y_hat, y_true)

    def compute_errors_online(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute errors between predicted and actual values.
        Adapted for online learning.

        Args:
            y_hat (np.ndarray): Predicted values of current batch.
            y_true (np.ndarray): Actual values of current batch.

        Returns:
            np.ndarray: Errors of current batch for each channel.
        """
        return self.detector.compute_errors_online(y_hat, y_true)

    def detect_anomalies(self, errors, y_true):
        """
        Detect anomalies based on the computed errors.
        Args:
            errors (np.ndarray): Computed errors.
            y_true (np.ndarray): True values.
        Returns:
            np.ndarray: Anomaly scores for each channel.
        """
        if self.config.dynamic_thresholding:
            anomaly_sequences = self.detector.detect_dynamic_anomalies_telemanom(errors, y_true)
            thresholded_scores = np.zeros_like(errors)

            for channel_idx, channel in enumerate(anomaly_sequences):
                for anomaly in channel:
                    start_idx, end_idx = anomaly['start_idx'], anomaly['end_idx']
                    thresholded_scores[start_idx:end_idx, channel_idx] = 1
            thresholds = np.zeros(thresholded_scores.shape)

            return thresholded_scores, thresholds
        return self.detector.rolling_window_thresholding(errors)
    
    
    def get_last_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, num_input_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # Pass through first LSTM layer
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # We only need the full sequence output here
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)

        # Pass through second LSTM layer
        # For the second LSTM, we only care about the output of the final time step
        # to make the prediction.
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # last_hidden_state shape: (batch_size, hidden_size)
        lstm_out, _ = self.lstm2(lstm_out)
        last_hidden_state = lstm_out[:, -1, :]
        last_hidden_state = self.dropout2(last_hidden_state)

        return last_hidden_state