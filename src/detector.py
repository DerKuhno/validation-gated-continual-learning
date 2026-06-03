import numpy as np
import pandas as pd
import more_itertools as mit
from torch import from_numpy, nn
from typing import List, Dict, Tuple
from tqdm import tqdm
from scipy.ndimage import maximum_filter1d

from src.experiment_config import ExperimentConfig

class Detector:
    """
    Handles the post-processing of model predictions to detect anomalies.
    This class re-implements the core logic from the original Telemanom's
    `errors.py` in a simplified and self-contained manner, fulfilling the
    role of both error computation and anomaly detection.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the Detector with experiment configurations.

        Args:
            config (ExperimentConfig): The configuration object for the experiment.
        """
        self.config = config
        self.model = config.model

        if self.model == 'telemanom':
            self.alpha = None
            self.last_weighted_sum = None # For online scenarios
            self.last_weight_sum = None # For online scenarios
        else:
            raise ValueError(f"Unsupported model type: {self.model}.")  

    def compute_errors(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        General entry point to compute errors. Dispatches to model-specific implementation.
        """
        method_name = f"compute_errors_{self.model}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(y_hat, y_true)
        raise ValueError(f"Unsupported model type: {self.model}")

    def compute_errors_telemanom(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Telemanom-specific smoothed error computation (original implementation).

        Args:
            y_hat (np.ndarray): Predicted values, shape (Timesteps, Channels).
            y_true (np.ndarray): Actual values, shape (Timesteps, Channels).

        Returns:
            np.ndarray: Smoothed errors (e_s), shape (Timesteps, Channels).
        """
        if y_hat.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} vs y_true {y_true.shape}")

        # 1. Calculate raw absolute error
        e = np.abs(y_hat - y_true)

        # 2. Apply exponential weighted moving average (smoothing)
        if self.config.dynamic_thresholding:
            smoothing_window_size = int(
                self.config.batch_size
                * self.config.smoothing_span
                * self.config.smoothing_percentile
            )
        else:
            smoothing_window_size = int(
                self.config.batch_size
                * self.config.rolling_smoothing_span
                * self.config.rolling_smoothing_percentile
            )

        e_s = pd.DataFrame(e).ewm(span=smoothing_window_size).mean().values

        # 3. Stabilize initial error values
        # For values at the beginning, EWM can be unstable. We replace them
        # with the mean of a slightly larger initial window.
        if len(e_s) > self.config.window_size * 2:
            e_s[:self.config.window_size] = np.mean(e_s[:self.config.window_size * 2], axis=0)
        
        return e_s

    def detect_anomalies(self, errors: np.ndarray, y_true: np.ndarray) -> List[List[Dict]]:
        """
        General entry point to detect anomalies. Dispatches to model-specific implementation.
        """
        method_name = f"detect_anomalies_{self.model}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(errors, y_true)
        raise ValueError(f"Unsupported model type for anomaly detection: {self.model}")

    def rolling_window_thresholding(self, errors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Statistical thresholding used in Validation-Gated Continual Learning.
        """

        N, C = errors.shape
        window_size = self.config.rolling_window_size
        thresholds = np.full((N, C), np.nan)
        anomalies = np.zeros((N, C), dtype=bool)

        for start_idx in tqdm(range(0, N, self.config.rolling_window_stride), desc=f"Threshold calculation"):
            # Determine the window range
            window_start = max(0, start_idx - window_size + 1)
            window_end = min(start_idx + self.config.rolling_window_stride, N)
            window = errors[window_start:window_end, :]  # shape: (win_len, C)

            # Sort and get lowest 99% per channel
            sorted_vals = np.sort(window, axis=0)
            cutoff_index = int(np.floor(self.config.rolling_calc_percentile * sorted_vals.shape[0]))
            lowest_99 = sorted_vals[:cutoff_index, :] if cutoff_index > 0 else sorted_vals

            # Mean and std per channel
            mean_val = np.mean(lowest_99, axis=0)
            std_val = np.std(lowest_99, axis=0, ddof=0)
            threshold = self.config.rolling_threshold_mean * mean_val +  self.config.rolling_threshold_std * std_val

            # Apply threshold to the whole batch
            batch_end = min(start_idx + self.config.rolling_window_stride, N)
            thresholds[start_idx:batch_end, :] = threshold
        
        thresholds[:2*self.config.window_size] = thresholds[2*self.config.window_size]

        anomalies = errors > thresholds

        # This padds 50 values to the front and back of an anomaly
        anomalies = maximum_filter1d(anomalies.astype(int), size=self.config.rolling_buffer+1, axis=0, mode="constant") > 0

        return anomalies, thresholds
    
    def detect_dynamic_anomalies_telemanom(self, errors: np.ndarray, y_true: np.ndarray) -> List[List[Dict]]:
        """
        Dynamic thresholding, used in the original Telemanom approach.

        Args:
            errors (np.ndarray): Smoothed errors from compute_errors(), shape (T, C).
            y_true (np.ndarray): Actual values, shape (T, C).

        Returns:
            List[List[Dict]]: A list where each element corresponds to a channel.
                              Each channel's element is a list of anomaly dictionaries,
                              e.g., [{'start_idx': s, 'end_idx': e, 'score': sc}, ...].
        """
        num_timesteps, num_channels = errors.shape
        all_channel_anomalies = []

        for c_idx in range(num_channels):
            e_s_chan = errors[:, c_idx]
            y_true_chan = y_true[:, c_idx]
            
            all_anom_scores_chan = []
            global_anoms = np.array([])
            
            # Use a sliding window to find anomalies, similar to `process_batches`
            for i in range(0, num_timesteps - self.config.window_size + 1, self.config.error_analysis_stride):
                start_idx = i
                end_idx = i + self.config.error_analysis_window_size
                is_first_window = (i == 0)

                # --- This block replaces the `ErrorWindow` class logic ---
                e_s_win, y_true_win = e_s_chan[start_idx:end_idx], y_true_chan[start_idx:end_idx]

                # Find thresholds for regular and inverted errors
                epsilon, _ = self._find_epsilon(e_s_win)
                mean_e_s_win = np.mean(e_s_win)
                e_s_inv_win = np.array([mean_e_s_win + (mean_e_s_win - e) for e in e_s_win])
                epsilon_inv, _ = self._find_epsilon(e_s_inv_win)

                # Find anomaly indices based on thresholds
                i_anom_win = self._compare_to_epsilon(e_s_win, y_true_win, epsilon, is_first_window)
                i_anom_inv_win = self._compare_to_epsilon(e_s_inv_win, y_true_win, epsilon_inv, is_first_window)

                # Prune and combine
                E_seq_win = self._prune_anoms(i_anom_win, e_s_win, global_anoms, start_idx)
                E_seq_inv_win = self._prune_anoms(i_anom_inv_win, e_s_inv_win, global_anoms, start_idx)
                i_anom_win = np.concatenate([np.arange(s, e + 1) for s, e in E_seq_win]) if E_seq_win else np.array([])
                i_anom_inv_win = np.concatenate([np.arange(s, e + 1) for s, e in E_seq_inv_win]) if E_seq_inv_win else np.array([])

                if len(i_anom_win) == 0 and len(i_anom_inv_win) == 0:
                    continue

                i_anom_total_win = np.sort(np.unique(np.append(i_anom_win, i_anom_inv_win))).astype(int)
                
                # Score anomalies and add to list with global indices
                anom_scores_win = self._score_anomalies(i_anom_total_win, e_s_win, e_s_inv_win, epsilon, epsilon_inv)
                for anom in anom_scores_win:
                    anom['start_idx'] += start_idx
                    anom['end_idx'] += start_idx
                    all_anom_scores_chan.append(anom)

                global_anoms = np.append(global_anoms, i_anom_total_win + start_idx)

            # --- Post-processing: Merge overlapping anomalies from different windows ---
            merged_anomalies = self.merge_anomalies(all_anom_scores_chan)
            all_channel_anomalies.append(merged_anomalies)
        
        return all_channel_anomalies

    def merge_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """
        Merges overlapping anomalies from different windows.

        Args:
            anomalies (List[Dict]): List of anomaly dictionaries with 'start_idx' and 'end_idx'.

        Returns:
            List[Dict]: Merged list of anomalies.
        """
        if not anomalies:
            return []

        # Sort by start index to make merging easier
        sorted_anoms = sorted(anomalies, key=lambda x: x['start_idx'])
        
        merged_anomalies = []
        if sorted_anoms:
            current_anom = sorted_anoms[0]
            for next_anom in sorted_anoms[1:]:
                # If the next anomaly overlaps or is adjacent to the current one
                if next_anom['start_idx'] <= current_anom['end_idx'] + 1:
                    # Merge them
                    current_anom['end_idx'] = max(current_anom['end_idx'], next_anom['end_idx'])
                    current_anom['score'] = max(current_anom['score'], next_anom['score'])
                else:
                    # Otherwise, the current merged anomaly is final
                    if self.config.prune_single_anoms:
                        if current_anom['end_idx'] == current_anom['start_idx']:
                            current_anom = next_anom
                            continue
                    merged_anomalies.append(current_anom)
                    current_anom = next_anom
            merged_anomalies.append(current_anom) # Add the last anomaly

        return merged_anomalies

    def _find_epsilon(self, e_s: np.ndarray) -> Tuple[float, float]:
        """Finds the optimal anomaly threshold (epsilon) for a window of errors."""
        sd_lim = self.config.z_end # original implementation: self.config.z_end = 12
        mean_e_s, sd_e_s = np.mean(e_s), np.std(e_s)
        if sd_e_s == 0: return mean_e_s, sd_lim

        max_score, best_z = -np.inf, sd_lim

        for z in np.arange(self.config.z_start, sd_lim, 0.5): #original implemention: self.config.z_start = 2.5
            epsilon = mean_e_s + (sd_e_s * z)
            pruned_e_s = e_s[e_s < epsilon]

            i_anom = np.argwhere(e_s >= epsilon).reshape(-1)
            if self.config.buffer_anoms:
                buffer = np.arange(1, self.config.error_buffer)
                i_anom = np.sort(np.concatenate((i_anom,
                                                np.array([i+buffer for i in i_anom])
                                                .flatten(),
                                                np.array([i-buffer for i in i_anom])
                                                .flatten())))
                i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
                i_anom = np.sort(np.unique(i_anom))
            if len(pruned_e_s) == len(e_s): break

            groups = [list(g) for g in mit.consecutive_groups(i_anom)]
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s if mean_e_s else 0
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s if sd_e_s else 0
            score = (mean_perc_decrease + sd_perc_decrease) / (len(E_seq)**2 + len(i_anom))
            
            if score >= max_score and len(E_seq) <= 5 and len(i_anom) < (len(e_s) * 0.5):
                max_score, best_z = score, z
        
        return mean_e_s + (sd_e_s * best_z), best_z

    def _compare_to_epsilon(self, e_s_win: np.ndarray, y_true_win: np.ndarray, epsilon: float, is_first_window: bool) -> np.ndarray:
        """Identifies anomaly indices by comparing errors to the threshold."""
        
        # Calculate statistics required for the check
        sd_e_s = np.std(e_s_win)
        sd_values = np.std(y_true_win)
        perc_high, perc_low = np.percentile(y_true_win, [95, 5])
        inter_range = perc_high - perc_low if perc_high > perc_low else 0
        max_e_s = np.max(e_s_win) if len(e_s_win) > 0 else 0

        # Check: scale of errors compared to values too small?
        if (
            not (
                sd_e_s > (0.05 * sd_values)
                or max_e_s > (0.05 * inter_range)
            )
            or len(e_s_win) < self.config.error_buffer
            or not max_e_s > self.config.min_error
        ):
            return np.array([]) # Return empty array if checks fail
        
        # Identify anomalies also considering their significance against the channel's range
        i_anom = np.argwhere(
            (e_s_win >= epsilon) & (e_s_win > 0.05 * inter_range)
        ).reshape(-1)

        if len(i_anom) == 0: return np.array([])
            
        buffer = np.arange(1, self.config.error_buffer + 1)
        i_anom = np.concatenate((i_anom, *[i_anom + b for b in buffer], *[i_anom - b for b in buffer]))
        i_anom = np.unique(i_anom[(i_anom < len(e_s_win)) & (i_anom >= 0)])

        if is_first_window: 
            i_anom = i_anom[i_anom >= self.config.num_to_ignore]
        else: 
            i_anom = i_anom[i_anom >= len(e_s_win) - self.config.batch_size]
        return i_anom.astype(int)

    def _prune_anoms(self, i_anom: np.ndarray, e_s_win: np.ndarray, global_anoms: np.ndarray, prior_idx: int) -> List[Tuple[int, int]]:
        """Prunes anomaly sequences that are not distinct enough."""
        if len(i_anom) == 0: return []
        
        groups = [list(g) for g in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        if not E_seq: return []
        E_seq_max = np.array([max(e_s_win[s:e+1]) for s, e in E_seq])

        # Shift current window indices to global positions
        batch_position = prior_idx  # Same as 'start_idx' for current window
        window_indices = np.arange(0, len(e_s_win)) + batch_position
        adj_i_anom = i_anom + batch_position

        # Exclude global anomalies (from other windows) and current anomalies
        non_anom_global = np.setdiff1d(window_indices, np.append(global_anoms, adj_i_anom))

        # Shift back to local window indices for indexing into e_s_win
        non_anom_indices = np.unique(non_anom_global - batch_position)

        non_anom_max = np.max(e_s_win[non_anom_indices]) if len(non_anom_indices) > 0 else -np.inf
        E_seq_max_sorted = np.append(np.sort(E_seq_max)[::-1], non_anom_max)
        i_to_remove = []
        for i in range(len(E_seq_max_sorted) - 1):
            # Avoid division by zero
            if E_seq_max_sorted[i] == 0: continue
            # If the relative drop to the next largest value is too small, mark for removal
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i+1]) / E_seq_max_sorted[i] < self.config.p_threshold:
                i_to_remove.extend(np.where(E_seq_max == E_seq_max_sorted[i])[0])
            else:
                # First significant drop found, stop pruning
                if self.config.prune_after_drop:
                    i_to_remove = []
                else:
                    break
        return [seq for i, seq in enumerate(E_seq) if i not in set(i_to_remove)]

    def _score_anomalies(self, i_anom_total: np.ndarray, e_s_win: np.ndarray, e_s_inv_win: np.ndarray, epsilon: float, epsilon_inv: float) -> List[Dict]:
        """Calculates scores for the final anomaly sequences in a window."""
        if len(i_anom_total) == 0: return []

        scores, groups = [], [list(g) for g in mit.consecutive_groups(i_anom_total)]
        denominator = (np.mean(e_s_win) + np.std(e_s_win)) or 1

        for seq in groups:
            if not seq: continue
            score_reg = max([abs(e_s_win[i] - epsilon) / denominator for i in range(seq[0], seq[-1] + 1)])
            score_inv = max([abs(e_s_inv_win[i] - epsilon_inv) / denominator for i in range(seq[0], seq[-1] + 1)])
            scores.append({'start_idx': seq[0], 'end_idx': seq[-1], 'score': max(score_reg, score_inv)})
            
        return scores
    

#
# --- Main block for verification ---
#
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("--- Running Detector Verification Script ---")

    # 1. Configuration
    config = ExperimentConfig()

    # 2. Generate Synthetic Data
    TIMESTEPS = 2000
    CHANNELS = 3
    ANOMALY_CHANNEL = 1
    ANOMALY_START = 1200
    ANOMALY_END = 1250
    ANOMALY_MAGNITUDE = 8.0

    print(f"\nGenerating synthetic data with {TIMESTEPS} timesteps and {CHANNELS} channels.")
    print(f"Injecting an anomaly in Channel {ANOMALY_CHANNEL} from t={ANOMALY_START} to t={ANOMALY_END}.")

    # Create a base signal (e.g., sine waves) with some noise
    time = np.linspace(0, 100, TIMESTEPS)
    y_true = np.zeros((TIMESTEPS, CHANNELS))
    for i in range(CHANNELS):
        y_true[:, i] = np.sin(time * (i + 1) * 0.5) + np.random.normal(0, 0.1, TIMESTEPS)

    # Create predictions (y_hat) that are similar but not perfect
    y_hat = y_true + np.random.normal(0, 0.2, y_true.shape)
    
    # Inject the anomaly into the true data
    y_true[ANOMALY_START:ANOMALY_END, ANOMALY_CHANNEL] += ANOMALY_MAGNITUDE
    
    print(f"Data generated. y_true shape: {y_true.shape}, y_hat shape: {y_hat.shape}")

    # 3. Instantiate and Run the Detector
    detector = Detector(config)
    print(f"\nDetector initialized with config: {vars(config)}")

    # Step A: Compute errors
    print("\nStep 1: Computing smoothed errors...")
    e_s = detector.compute_errors(y_hat, y_true)
    print(f"Smoothed errors 'e_s' computed. Shape: {e_s.shape}")

    # Step B: Detect anomalies
    print("\nStep 2: Detecting anomalies from smoothed errors...")
    detected_anomalies = detector.detect_anomalies(e_s, y_true)

    # 4. Display Results
    print("\n--- Detection Results ---")
    found_any = False
    for i, channel_anomalies in enumerate(detected_anomalies):
        if not channel_anomalies:
            print(f"Channel {i}: No anomalies detected.")
        else:
            found_any = True
            print(f"Channel {i}: Found {len(channel_anomalies)} anomaly sequence(s):")
            for anom in channel_anomalies:
                print(f"  - Start: {anom['start_idx']}, End: {anom['end_idx']}, Score: {anom['score']:.4f}")
    
    if not found_any:
        print("\nVerification FAILED: No anomalies were detected.")
    else:
        print("\nVerification PASSED: Anomaly sequences were found.")