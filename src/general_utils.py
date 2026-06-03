import logging
from tqdm import tqdm
import numpy as np
import time
import torch
import hashlib
import json
from typing import Any, Dict, TYPE_CHECKING
from pathlib import Path

from src.telemanom_model import TelemanomModel
if TYPE_CHECKING:
    from src.dataset_manager import DatasetManager
from src.experiment_config import ExperimentConfig


def initial_training(model: TelemanomModel, dataset_manager: "DatasetManager", config: ExperimentConfig):
    logging.info("Starting the initial training...")

    train_loader = dataset_manager.get_dataloader('train', config.batch_size, shuffle=True)
    val_loader = dataset_manager.get_dataloader('val', config.val_batch_size, shuffle=False)
    
    max_training_steps = min(len(train_loader), config.max_steps_per_epoch)
    best_val_loss = float('inf')
    patience_counter = 0
    
    ckpt_path = config.results_path / "best_model.pth"
    
    # Loop over epochs
    for epoch in range(config.num_epochs):
        logging.info(f"Epoch [{epoch+1}/{config.num_epochs}]")

        training_steps = 0
        model.train()
        # Progress bar for batches
        progress_bar = tqdm(train_loader, total=max_training_steps, desc=f"Epoch {epoch+1} training")

        start_time = time.time()
        for batch in progress_bar:
            inputs, targets = batch
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            loss = model.train_step(inputs, targets)
            # Update tqdm with current loss
            progress_bar.set_postfix({"loss": f"{loss.cpu().item():.4f}"})
            training_steps += 1
            if training_steps > max_training_steps:
                break
        end_time = time.time()
        logging.info(f"Epoch {epoch+1} training completed in {end_time - start_time:.2f} seconds.")

        # Validation logic with early stopping (patience and delta)
        val_losses = []
        model.eval()

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} validation")

        start_time = time.time()
        with torch.no_grad():
            for val_batch in val_progress_bar:
                val_inputs, val_targets = val_batch
                val_inputs, val_targets = val_inputs.to(config.device), val_targets.to(config.device)

                val_loss = model.validation_step(val_inputs, val_targets)

                val_progress_bar.set_postfix({"val_loss": f"{val_loss.cpu().item():.4f}"})
                val_losses.append(val_loss.cpu().item())
            avg_val_loss = np.mean(val_losses)

        end_time = time.time()
        logging.info(f"Epoch {epoch+1} validation completed in {end_time - start_time:.2f} seconds.")
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if best_val_loss - avg_val_loss > config.early_stopping_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logging.info(f"New best validation loss: {best_val_loss:.4f}.")
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{config.early_stopping_patience}.")

        if patience_counter >= config.early_stopping_patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
    model.load_state_dict(torch.load(ckpt_path))
    model.to(config.device)

    logging.info("Initial training completed.")

def get_cache_path(
    base_path: Path,
    config: ExperimentConfig,
    context: Dict[str, Any],
    prefix: str = "dataset_cache",
) -> Path:
    """
    Generates a unique cache file path based on the configuration and context.
    """
    hasher = hashlib.sha256()

    # Add relevant config to hash
    config_dict = {
        "collection": config.collection,
        "dataset": config.dataset,
        "input_channels": config.input_channels,
        "target_channels": config.target_channels,
        "window_size": config.window_size,
        "prediction_window_size": config.prediction_window_size,
        "validation_split_ratio": config.validation_split_ratio,
        "model": config.model,
    }
    hasher.update(json.dumps(config_dict, sort_keys=True).encode())

    # Add context to hash
    context_str = json.dumps(context, sort_keys=True, default=str)
    hasher.update(context_str.encode())

    return base_path / f"{prefix}_{hasher.hexdigest()}.pkl"
