import logging
from tqdm import tqdm
import numpy as np
from torch import no_grad
import time

def calculate_val_loss(model, eval_dataloader, device, epoch):
    high_validation_loss = 0
    
    val_losses = []
    model.eval()

    val_progress_bar = tqdm(eval_dataloader, desc=f"CL Epoch {epoch+1} validation")

    start_time = time.time()
    with no_grad():
        for val_batch in val_progress_bar:
            val_inputs, val_targets = val_batch
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

            val_loss = model.validation_step(val_inputs, val_targets)

            val_progress_bar.set_postfix({"val_loss": f"{val_loss.cpu().item():.4f}"})
            val_losses.append(val_loss.cpu().item())
            if val_loss > 1:
                high_validation_loss +=1
        avg_val_loss = np.mean(val_losses)
    end_time = time.time()
    logging.info(f"Experience epoch {epoch+1} validation completed in {end_time - start_time:.2f} seconds.")
    logging.info(f"Validation Loss: {avg_val_loss:.4f}, batches with high loss: {high_validation_loss}")
    return avg_val_loss

def get_buffer_iterator(dataloader):
        while True:                # Infinite loop
            for batch in dataloader:
                yield batch  # Yield one batch at a time