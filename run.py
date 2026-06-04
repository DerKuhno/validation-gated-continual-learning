from pathlib import Path
import logging
import atexit
import time

# Assuming the two files above are in the same directory
from src.dataset_manager import DatasetManager
from src.telemanom_model import TelemanomModel
from src.experiment_config import ExperimentConfig
from src.Main_loop import run, set_random_seeds

######## Iter over Experiment Parameters ########
import itertools
hyperparameter_grid = {
}


def main(hyperparams: dict):
    """
    This function initialises the experiment.
    All hyperparameter changes should be done here.
    
    """
    
    root = Path(__file__).parent
    # Skips pretraining, if a pretrained model is provided under: "results/16Weeks-Baseline-Seed-{seed}/best_model.pth"
    # pretrained_model_path = root / "results/16Weeks-Baseline"

    config = ExperimentConfig(
        ### Train/Val/Test set ###
        collection = "ESA-Mission1",
        dataset = "84_months",
        hyperparameter_search=hyperparams,
        # pre_trained_model_path = pretrained_model_path,
    )
    init_logging(config.results_path)

    if 'seed' in config.hyperparameter_search.keys():
        print(f"--- Initialized with seed: {config.hyperparameter_search['seed']} ---")
        logging.info(f"--- Initialized with seed: {config.hyperparameter_search['seed']} ---")
    print(f"--- Pretrained with: {config.pre_trained_model_path} ---")
    logging.info(f"--- Pretrained with: {config.pre_trained_model_path} ---")
    logging.info(f"Running experiment with config:\n{config}")
    print(f"--- Initializing and using {config.collection}/{config.dataset} data ---")

    set_random_seeds(config)

    logging.info(f"Model is initializing...")
    if config.model == "telemanom":
        model = TelemanomModel(config).to(config.device)
    logging.info(f"Model initialized!")


    
    logging.info(f"Datasetmanager is initializing...")
    dataset_manager_list = [DatasetManager(
        config=config,
        include_test=True,  # True: loads test data
    )]
    logging.info(f"Datasetmanager initialized!")


    try:
        run(
            model=model,
            config=config,
            dataset_manager_list=dataset_manager_list
        )
        
        logging.info(f"Experiment done successfully! Logging is closed")
    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}")
        raise e



def init_logging(results_path: Path | str):
    """
    example usage:
    logging.debug("This is debug-level info (for developers)")
    logging.info("This is an info message (for normal progress)")
    logging.warning("This is a warning (something unexpected happened)")
    logging.error("This is an error (a recoverable failure)")
    logging.critical("This is critical (something is badly broken!)")
    """
    results_path = Path(results_path)
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    log_file = results_path / "experiment.log"

    logging.basicConfig(
        filename=log_file,
        filemode="a",  # Append mode
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG  # Minimum level to log. Other options: logging.DEBUG, logging.INFO, logging.WARNING
    )
    logging.info(f"Logger initialized. Writing logs to {log_file}")


if __name__ == "__main__":
    _start_time = time.time()

    def _log_total_runtime():
                total_time = time.time() - _start_time
                print(f"\nTotal run time: {total_time:.2f} seconds")
                logging.info(f"Total run time: {total_time:.2f} seconds")

    atexit.register(_log_total_runtime)

    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())

    if hyperparameter_grid:
        for values in itertools.product(*param_values):
            hyperparams = dict(zip(param_names, values))

            main(hyperparams)
    else: main({})