# tests/test_detection.py

import pandas as pd
from plotsense_anomaly.detection import zscore_anomaly_detection

def test_simple_anomaly_detection():
    data = [10, 12, 11, 200, 13]
    df = zscore_anomaly_detection(data, threshold=1.5)
    anomalies = df[df["anomaly"]]["value"].tolist()
    assert 200 in anomalies  # 200 should be flagged as an anomaly
