import pandas as pd
import numpy as np

def zscore_anomaly_detection(data, threshold=1.5):
    """
    Detect anomalies in a numeric dataset using Z-score.
    
    Parameters:
        data (list or pd.Series): Input numeric data.
        threshold (float): Z-score cutoff for anomalies (default=2.0).
    
    Returns:
        pd.DataFrame with columns [value, zscore, anomaly]
    """
    df = pd.DataFrame({"value": data})
    mean = df["value"].mean()
    std = df["value"].std()
    
    df["zscore"] = (df["value"] - mean) / std
    df["anomaly"] = np.abs(df["zscore"]) > threshold
    return df
