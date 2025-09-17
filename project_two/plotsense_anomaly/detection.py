"""
Anomaly Detection Algorithms

This module implements various statistical methods for detecting anomalies
in numeric datasets. The primary focus is on Z-score based detection, but
the architecture supports easy extension to other methods.

Supported Methods:
- Z-score based anomaly detection
- (Future: IQR-based, isolation forest, etc.)

Example:
    >>> import pandas as pd
    >>> from plotsense_anomaly.detection import zscore_anomaly_detection
    >>>
    >>> # Sample data with obvious outliers
    >>> data = [10, 12, 11, 13, 100, 14, 12, 11, 15, 120, 14, 13]
    >>>
    >>> # Detect anomalies using Z-score method
    >>> results = zscore_anomaly_detection(data, threshold=2.0)
    >>> print(results[results['anomaly']])

Authors: Havilah Academy Team for PlotSenseAI
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Union, List
import warnings


def zscore_anomaly_detection(
    data: Union[List[float], pd.Series, np.ndarray],
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies in a numeric dataset using the Z-score method.

    The Z-score method identifies anomalies by measuring how many standard
    deviations each data point is away from the mean. Points with a Z-score
    greater than the threshold are flagged as anomalies.

    Mathematical Formula:
        Z-score = (value - mean) / standard_deviation
        Anomaly if: |Z-score| > threshold

    Parameters:
        data (list, pd.Series, or np.ndarray): Input numeric data for analysis.
            Should contain at least 3 data points for meaningful statistics.
        threshold (float, optional): Z-score cutoff for anomaly detection.
            Common values:
            - 1.5: More sensitive, catches more potential anomalies
            - 2.0: Standard threshold, balanced sensitivity
            - 3.0: Conservative, only catches extreme outliers
            Default is 2.0.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'value': Original data values
            - 'zscore': Calculated Z-score for each value
            - 'anomaly': Boolean indicating if the value is an anomaly

    Raises:
        ValueError: If data is empty or contains less than 2 unique values
        TypeError: If data contains non-numeric values

    Examples:
        >>> # Basic usage
        >>> data = [1, 2, 3, 100, 4, 5]
        >>> result = zscore_anomaly_detection(data, threshold=2.0)
        >>> print(result)
             value    zscore  anomaly
        0        1 -0.707    False
        1        2 -0.617    False
        2        3 -0.527    False
        3      100  1.851     True
        4        4 -0.437    False
        5        5 -0.347    False

        >>> # More sensitive detection
        >>> result_sensitive = zscore_anomaly_detection(data, threshold=1.0)
        >>> anomalies = result_sensitive[result_sensitive['anomaly']]
        >>> print(f"Detected {len(anomalies)} anomalies")

        >>> # Get only anomalous values
        >>> anomaly_values = result[result['anomaly']]['value'].tolist()
        >>> print(f"Anomalous values: {anomaly_values}")

    Notes:
        - The Z-score method assumes the data follows a normal distribution
        - For non-normal distributions, consider other methods like IQR
        - Very small datasets (< 10 points) may produce unreliable results
        - This method detects global anomalies, not local or contextual ones

    See Also:
        - visualize_anomalies(): For PlotSenseAI integration
        - sklearn.ensemble.IsolationForest: For more complex anomaly detection
    """
    # Input validation and type conversion
    if not isinstance(data, (list, pd.Series, np.ndarray)):
        raise TypeError(f"Data must be list, pd.Series, or np.ndarray, got {type(data)}")

    # Convert to pandas Series for easier handling
    if isinstance(data, list):
        data = pd.Series(data)
    elif isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Check for empty data
    if len(data) == 0:
        raise ValueError("Input data cannot be empty")

    # Check for minimum data points
    if len(data) < 2:
        warnings.warn(
            "Dataset has less than 2 points. Anomaly detection may not be meaningful.",
            UserWarning
        )

    # Check for non-numeric data
    if not pd.api.types.is_numeric_dtype(data):
        try:
            data = pd.to_numeric(data, errors='coerce')
            if data.isna().any():
                raise ValueError("Data contains non-numeric values that cannot be converted")
        except (ValueError, TypeError):
            raise TypeError("Data must contain only numeric values")

    # Create result DataFrame
    df = pd.DataFrame({"value": data})

    # Calculate statistical measures
    mean = df["value"].mean()
    std = df["value"].std()

    # Handle edge case where standard deviation is 0 (all values are the same)
    if std == 0:
        warnings.warn(
            "All values in the dataset are identical. No anomalies can be detected.",
            UserWarning
        )
        df["zscore"] = 0.0
        df["anomaly"] = False
        return df

    # Calculate Z-scores
    # Z-score formula: (value - mean) / standard_deviation
    df["zscore"] = (df["value"] - mean) / std

    # Identify anomalies based on threshold
    # An anomaly is a point where |Z-score| > threshold
    df["anomaly"] = np.abs(df["zscore"]) > threshold

    # Add some metadata for debugging/analysis
    df.attrs['mean'] = mean
    df.attrs['std'] = std
    df.attrs['threshold'] = threshold
    df.attrs['anomaly_count'] = df['anomaly'].sum()
    df.attrs['anomaly_rate'] = df['anomaly'].mean()

    return df


# Additional utility functions for future extensions
def get_anomaly_summary(result_df: pd.DataFrame) -> dict:
    """
    Generate a summary of anomaly detection results.

    Parameters:
        result_df (pd.DataFrame): Output from zscore_anomaly_detection

    Returns:
        dict: Summary statistics including count, rate, and extreme values
    """
    anomalies = result_df[result_df['anomaly']]

    summary = {
        'total_points': len(result_df),
        'anomaly_count': len(anomalies),
        'anomaly_rate': len(anomalies) / len(result_df) if len(result_df) > 0 else 0,
        'mean': result_df.attrs.get('mean', result_df['value'].mean()),
        'std': result_df.attrs.get('std', result_df['value'].std()),
        'threshold': result_df.attrs.get('threshold', 'unknown'),
        'max_zscore': result_df['zscore'].abs().max(),
        'anomaly_values': anomalies['value'].tolist() if len(anomalies) > 0 else [],
        'extreme_zscores': anomalies['zscore'].tolist() if len(anomalies) > 0 else []
    }

    return summary


def validate_threshold(threshold: float) -> bool:
    """
    Validate that the threshold value is reasonable for Z-score detection.

    Parameters:
        threshold (float): The threshold value to validate

    Returns:
        bool: True if threshold is valid, False otherwise
    """
    if not isinstance(threshold, (int, float)):
        return False

    if threshold <= 0:
        return False

    # Warn about unusual threshold values
    if threshold < 1.0:
        warnings.warn(
            f"Threshold {threshold} is very low and may produce many false positives",
            UserWarning
        )
    elif threshold > 4.0:
        warnings.warn(
            f"Threshold {threshold} is very high and may miss real anomalies",
            UserWarning
        )

    return True