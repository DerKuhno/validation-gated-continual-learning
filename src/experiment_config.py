from dataclasses import dataclass, field, fields
from typing import List, Union
from pathlib import Path
from datetime import datetime
import torch

from src.metrics.ESA_ADB_metrics import ESAScores

DEFAULT_MISSION1_CHANNELS = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"] # Subset
DEFAULT_MISSION1_CHANNELS_INDICES = [35, 36, 37, 38, 39, 40]
DEFAULT_MISSION2_CHANNELS = ["channel_18", "channel_19", "channel_20", "channel_21", "channel_22", "channel_23", "channel_24", "channel_25", "channel_26", "channel_27", "channel_28"]
DEFAULT_MISSION2_CHANNELS_INDICES = [11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]
DEFAULT_CHANNELS = {
    "ESA-Mission1": DEFAULT_MISSION1_CHANNELS,
    "ESA-Mission2": DEFAULT_MISSION2_CHANNELS,
}

ROOT = Path(__file__).parent.parent

DEFAULT_BETA = 0.5
DEFAULT_METRICS = [
                        ESAScores(betas=DEFAULT_BETA, select_labels={"Category": ["Rare Event", "Anomaly"]}),
                        ESAScores(betas=DEFAULT_BETA, select_labels={"Category": ["Anomaly"]}),
                    ]

def get_current_timestamp() -> str:
    """Returns the current timestamp as a formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class AlgorithmParams:
    def __init__(self):
        pass


@dataclass
class ExperimentConfig:
    """Encodes all hyperparameters of an experiment."""
    experiment_id: str = field(default_factory=get_current_timestamp)
    results_path: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_reconstructions: bool = True
    save_errors: bool = True
    save_unprocessed_errors: bool = True
    save_anomalies: bool = True
    save_thresholds: bool = False # only meaningful for rolling window thresholding
    pre_trained_model_path: Path = None


    # Dataset parameters
    collection: str = "ESA-Mission1" # "ESA-Mission1" "ESA-Mission2"
    dataset: Union[str, List[str]] = None
    validation_split_ratio: float = 0.2         # (1-alpha)
    input_channels: List[str] = None
    target_channels: List[str] = None
    input_channel_indices: List[int] = None
    target_channel_indices: List[int] = None
    window_size: int = 250
    stride: int = None
    use_cache: bool = True

    # Model parameters
    model: str = "telemanom"
    prediction_window_size: int = 10
    lstm_layers: List[int] = field(default_factory=lambda: [80, 80])
    dropout: float = 0.3

    # Training parameters
    num_epochs: int = 1000
    max_steps_per_epoch: int = 1000
    early_stopping_patience: int = 20
    early_stopping_delta: float = 0.0003
    batch_size: int = 70
    val_batch_size: int = 70
    test_batch_size: int = 70
    lr: float = 0.001
    loss_function: str = "mse"
    optimizer: str = "adam"

    # Continual Learning parameters
    continual_learning: bool = False
    fixed_epochs: bool = False
    fixed_num_epochs: int = 1
    last_16_weeks: bool = True                  # uses the last 16 weeks of the data as experience 0
    online_learning: bool = False
    cl_method: str = "agem"
    replay_strat: str = "none"
    batches_per_exp: int = None

    # Metric parameters
    metrics: list = field(default_factory=lambda: DEFAULT_METRICS)
    error_score_cutoff: float = 95              # trimmed MAE


    # Detector parameters
    # Dynamic Thresholding
    z_start: bool = 2.5 # Dynamic thresholding: 2.5
    z_end: int = 12 # Original implementation: 12
    smoothing_span: int = 30
    smoothing_percentile: float = 0.05
    p_threshold: float = 0.13
    error_buffer: int = 100
    error_analysis_window_size: int = None
    error_analysis_stride: int = None
    num_to_ignore: int = None 
    min_error: float = 0.0
    buffer_anoms: bool = True #TODO: delete
    prune_after_drop: bool = True #TODO: delete
    prune_single_anoms: bool = True #TODO: delete
    # Parametric Thresholding # values from Hyperparametersearch: 346
    dynamic_thresholding: bool = False
    rolling_threshold_std: float = 8.5
    rolling_threshold_mean: float = 1.3
    rolling_smoothing_percentile: float = 0.019
    rolling_window_size: int = 5000
    rolling_smoothing_span: int = 30
    rolling_calc_percentile: int = 0.99
    rolling_window_stride: int = 70
    rolling_buffer: int = 200


    # Algorithm specific params go into this dict
    method_params: AlgorithmParams = field(default_factory=lambda: AlgorithmParams())

    # Reproducibility
    seed_random: int = 42
    seed_np: int = 42
    seed_torch: int = 42

    # Hyperparameter search
    hyperparameter_search: dict = field(default_factory=lambda: {})
    hyperparameter_search_split: bool = False

    def __post_init__(self):
        # Apply top-level config overrides from hyperparameter_search before derived fields are computed
        _config_fields = {f.name for f in fields(self)}
        for key, value in self.hyperparameter_search.items():
            if key in _config_fields:
                setattr(self, key, value)

        self.results_path = ROOT / "results" / self.experiment_id

        # Using the correct seed
        if 'seed' in self.hyperparameter_search.keys():
            if self.pre_trained_model_path:
                self.pre_trained_model_path = Path(str(self.pre_trained_model_path) + f"-Seed-{self.hyperparameter_search['seed']}/best_model.pth")
            self.seed_random = self.hyperparameter_search['seed']
            self.seed_np = self.hyperparameter_search['seed']
            self.seed_torch = self.hyperparameter_search['seed']
        elif self.pre_trained_model_path:
            self.pre_trained_model_path = self.pre_trained_model_path / "best_model.pth"

        self.num_to_ignore = self.window_size * 2
        self.error_analysis_window_size = self.smoothing_span * self.batch_size
        self.error_analysis_stride = self.batch_size

        if self.collection in ["ESA-Mission1", "ESA-Mission2"]:
            self.labels_csv_path = ROOT / f"data/{self.collection}/labels.csv"
            self.anomaly_types_path = ROOT / f"data/{self.collection}/anomaly_types.csv"
            self.channels_path = ROOT / f"data/{self.collection}/channels.csv"

            if self.collection == "ESA-Mission1":
                self.dataset = "84_months" if self.dataset is None else self.dataset
                self.input_channels = DEFAULT_MISSION1_CHANNELS if self.input_channels is None else self.input_channels
                self.target_channels = DEFAULT_MISSION1_CHANNELS if self.target_channels is None else self.target_channels
                self.input_channel_indices = DEFAULT_MISSION1_CHANNELS_INDICES if self.input_channel_indices is None else self.input_channel_indices
                self.target_channel_indices = DEFAULT_MISSION1_CHANNELS_INDICES if self.target_channel_indices is None else self.target_channel_indices
            else: # ESA-Mission2
                self.dataset = "21_months" if self.dataset is None else self.dataset
                self.input_channels = DEFAULT_MISSION2_CHANNELS if self.input_channels is None else self.input_channels
                self.target_channels = DEFAULT_MISSION2_CHANNELS if self.target_channels is None else self.target_channels
                # For ESA-Mission2, we assume the indices are the same as the channels order in DEFAULT_MISSION2_CHANNELS
                self.input_channel_indices = DEFAULT_MISSION2_CHANNELS_INDICES if self.input_channel_indices is None else self.input_channel_indices
                self.target_channel_indices = DEFAULT_MISSION2_CHANNELS_INDICES if self.target_channel_indices is None else self.target_channel_indices

            self.test_dataset_path = ROOT / f"data/preprocessed/multivariate/{self.collection}-semi-supervised/{self.dataset}.test.csv"

        else:   
            raise ValueError(f"The collection isn't availble. Check if {self.collection} is in ['ESA-Mission1', 'ESA-Mission2']")

        if self.collection == "ESA-Mission1":
            self.batches_per_exp = 4608 # 4608 * 70 batch samples * 0.5 min = 16 Weeks * 7 Days * 24 h * 60 min
        elif self.collection == "ESA-Mission2":
            self.batches_per_exp = 7680 # 7680 * 70 batch samples * 0.3 min = 16 Weeks * 7 Days * 24 h * 60 min