"""
PlotSenseAI Anomaly Detection Plugin

A comprehensive anomaly detection plugin that integrates seamlessly with PlotSenseAI
for automated visualization and explanation of anomalous data patterns.

This package provides:
- Statistical anomaly detection algorithms (Z-score based)
- PlotSenseAI integration for automatic visualization
- Extensible architecture for custom detection methods
- Comprehensive testing suite

Example Usage:
    >>> from plotsense_anomaly import zscore_anomaly_detection
    >>> from plotsense_anomaly.visualization import visualize_anomalies
    >>>
    >>> # Detect anomalies in your data
    >>> data = [1, 2, 3, 100, 4, 5, 6]
    >>> results = zscore_anomaly_detection(data, threshold=2.0)
    >>>
    >>> # Visualize with PlotSenseAI
    >>> plot = visualize_anomalies(results)
    >>> plot.show()

Authors: Havilah Academy Team for PlotSenseAI
Version: 1.0.0
License: MIT
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Havilah Academy Team"
__email__ = "support@havilahacademy.org"
__license__ = "MIT"

# Import main functions for easy access
from .detection import zscore_anomaly_detection
from .visualization import visualize_anomalies

# Define what gets imported with "from plotsense_anomaly import *"
__all__ = [
    "zscore_anomaly_detection",
    "visualize_anomalies",
]
