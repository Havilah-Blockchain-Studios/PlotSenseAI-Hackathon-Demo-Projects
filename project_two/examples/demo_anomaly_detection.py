# examples/demo_anomaly_detection.py

import pandas as pd
import matplotlib.pyplot as plt

# from plotsense_anomaly.detection import zscore_anomaly_detection
# from plotsense_anomaly.visualization import visualize_anomalies

# # Generate sample data with anomalies
# data = pd.Series([10, 12, 11, 13, 150, 14, 12, 11, 15, 250, 14, 13])

# # Run anomaly detection
# result = zscore_anomaly_detection(data, threshold=1.0)
# print(result)

# # Visualize anomalies with PlotSense
# plot = visualize_anomalies(result, n=1)
# plt.show()

from plotsense_anomaly.detection import zscore_anomaly_detection
from plotsense_anomaly.visualization import visualize_anomalies

# Example dataset with obvious outliers
data = [10, 12, 11, 13, 100, 14, 12, 11, 15, 120, 14, 13]

# Run detection (lower threshold if needed)
df = zscore_anomaly_detection(data, threshold=1.0)

# Print full results
print(df)

# Print only anomalies
anomalies = df[df["anomaly"]]["value"].tolist()
print(f"\nAnomalies detected: {anomalies}\n")

# Visualize anomalies with PlotSense
visualize_anomalies(df)
plt.show()
