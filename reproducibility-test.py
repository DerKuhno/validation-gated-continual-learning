import pandas as pd
from pathlib import Path
import numpy as np

path_metrics1 = Path("results/reproducibility1")
path_metrics2 = Path("results/reproducibility2")

# Load both DataFrames
df1 = pd.read_csv(path_metrics1 / "metrics.csv")
df2 = pd.read_csv(path_metrics2 / "metrics.csv")

# Check if they are exactly equal
are_equal = df1.equals(df2)

print("Are the Metrics exactly the same?", are_equal)

# Check Errors
errors1 = np.loadtxt(path_metrics1 / "errors.csv", delimiter=",")
errors2 = np.loadtxt(path_metrics2 / "errors.csv", delimiter=",")

print("Difference in the errors: ", np.sum(np.abs(errors1-errors2)))

# Check Predictions
pred1 = np.loadtxt(path_metrics1 / "anomaly_prediction.csv", delimiter=",")
pred2 = np.loadtxt(path_metrics2 / "anomaly_prediction.csv", delimiter=",")

print("Difference in the predictions: ", np.sum(np.abs(pred1-pred2)))
