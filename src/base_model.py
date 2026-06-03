import torch
import torch.nn as nn
import numpy as np
import logging

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def aggregate_predictions(self, y_hat, method='first'):
        raise NotImplementedError

    def compute_errors(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute_errors_online(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def detect_anomalies(self, errors, y_true):
        raise NotImplementedError

    def reset_weights(self):
        """
        Reinitialize model parameters.
        """
        print("Reinitializing the model")
        logging.info("Reinitializing the model")
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def get_last_latent(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Loss value after the training step.
        """
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def validation_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Loss value after the validation step.
        """
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss
    
    def calc_error_percentile_score(self, unprocessed_errors) -> dict:
        mean_errors = []
        for ch in range(unprocessed_errors.shape[1]):
            channel_data = unprocessed_errors[:, ch]

            # find percentile cutoff
            cutoff = np.percentile(channel_data, self.config.error_score_cutoff)

            # keep only lower percentile values
            lower_vals = channel_data[channel_data <= cutoff]

            # compute mean (error or just mean of these values)
            mean_err = np.mean(lower_vals)

            mean_errors.append(mean_err)

        mean_errors = np.array(mean_errors)
        return {f"mean_{self.config.error_score_cutoff}percentile_error": np.mean(mean_errors), f"mean_{self.config.error_score_cutoff}_channel_errors": mean_errors}
