"""
PlotSenseAI Integration for Anomaly Detection

This module provides seamless integration between anomaly detection results
and PlotSenseAI's visualization capabilities. It automatically generates
appropriate visualizations for anomaly detection results and provides
explanations for detected patterns.

Features:
- Automatic visualization recommendation for anomaly data
- Support for different plot types and styles
- Integration with PlotSenseAI's explanation engine
- Customizable visualization parameters

Example:
    >>> from plotsense_anomaly.detection import zscore_anomaly_detection
    >>> from plotsense_anomaly.visualization import visualize_anomalies
    >>>
    >>> # Detect anomalies
    >>> data = [1, 2, 3, 100, 4, 5, 6]
    >>> results = zscore_anomaly_detection(data, threshold=2.0)
    >>>
    >>> # Visualize with PlotSenseAI
    >>> plot = visualize_anomalies(results, n=3)
    >>> plot.show()

Authors: Havilah Academy Team for PlotSenseAI
License: MIT
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import warnings

try:
    import plotsense as ps
    from plotsense import plotgen, recommender, explainer
    PLOTSENSE_AVAILABLE = True
except ImportError:
    PLOTSENSE_AVAILABLE = False
    warnings.warn(
        "PlotSense is not available. Visualization features will be limited.",
        ImportWarning
    )


def visualize_anomalies(
    df: pd.DataFrame,
    n: int = 3,
    plot_index: int = 0,
    show_plot: bool = True,
    return_recommendations: bool = False
) -> Union[plt.Figure, tuple]:
    """
    Visualize anomaly detection results using PlotSenseAI.

    This function takes the output from anomaly detection algorithms and
    automatically generates appropriate visualizations using PlotSenseAI's
    recommendation engine. It highlights anomalous points and provides
    clear visual distinction between normal and anomalous data.

    Parameters:
        df (pd.DataFrame): Anomaly detection results with columns:
            - 'value': Original data values
            - 'zscore': Z-score for each value
            - 'anomaly': Boolean indicating anomalies
        n (int, optional): Number of visualization recommendations to generate.
            Default is 3. Higher values provide more options but take longer.
        plot_index (int, optional): Index of recommendation to use for plotting.
            Default is 0 (first recommendation). Must be < n.
        show_plot (bool, optional): Whether to display the plot immediately.
            Default is True.
        return_recommendations (bool, optional): Whether to return the
            recommendations DataFrame along with the plot. Default is False.

    Returns:
        plt.Figure or tuple: If return_recommendations is False, returns the
            matplotlib Figure object. If True, returns tuple of (Figure,
            recommendations DataFrame).

    Raises:
        ValueError: If df doesn't contain required columns or plot_index >= n
        ImportError: If PlotSense is not available
        TypeError: If input parameters have incorrect types

    Examples:
        >>> # Basic usage
        >>> from plotsense_anomaly.detection import zscore_anomaly_detection
        >>> data = [10, 12, 11, 13, 100, 14, 12, 11, 15, 120, 14, 13]
        >>> results = zscore_anomaly_detection(data, threshold=2.0)
        >>> plot = visualize_anomalies(results)

        >>> # Get multiple recommendations
        >>> plot, recs = visualize_anomalies(
        ...     results,
        ...     n=5,
        ...     return_recommendations=True
        ... )
        >>> print(recs)

        >>> # Use different plot style
        >>> plot = visualize_anomalies(results, plot_index=1)

        >>> # Don't show plot immediately
        >>> plot = visualize_anomalies(results, show_plot=False)
        >>> # ... customize plot ...
        >>> plot.show()

    Notes:
        - PlotSenseAI automatically chooses the best visualization type
        - Anomalies are typically highlighted in red
        - Normal points are shown in blue or default colors
        - The function works with any anomaly detection algorithm output
          that follows the expected DataFrame format

    See Also:
        - zscore_anomaly_detection(): For generating the input DataFrame
        - get_anomaly_explanation(): For AI-generated explanations
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df)}")

    # Check required columns
    required_columns = ['value', 'zscore', 'anomaly']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Validate parameters
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Parameter 'n' must be positive integer, got {n}")

    if not isinstance(plot_index, int) or plot_index < 0:
        raise ValueError(f"Parameter 'plot_index' must be non-negative integer, got {plot_index}")

    if plot_index >= n:
        raise ValueError(f"plot_index ({plot_index}) must be less than n ({n})")

    # Check PlotSense availability
    if not PLOTSENSE_AVAILABLE:
        return _fallback_visualization(df, show_plot)

    try:
        # Prepare data for PlotSenseAI
        # Add an index column for better visualization
        viz_data = df.copy()
        viz_data['index'] = range(len(df))

        # Get visualization recommendations from PlotSenseAI
        # PlotSenseAI analyzes the data structure and suggests optimal visualizations
        suggestions = recommender(viz_data, n=n)

        # Generate the plot using the selected recommendation
        plot = plotgen(viz_data, plot_index, suggestions)

        # Enhance the plot with anomaly-specific styling
        _enhance_anomaly_plot(plot, df)

        # Display plot if requested
        if show_plot:
            plot.show()

        # Return based on user preference
        if return_recommendations:
            return plot, suggestions
        else:
            return plot

    except Exception as e:
        warnings.warn(
            f"PlotSense visualization failed: {e}. Falling back to basic plot.",
            UserWarning
        )
        return _fallback_visualization(df, show_plot)


def get_anomaly_explanation(
    df: pd.DataFrame,
    plot: Optional[plt.Figure] = None,
    context: str = "anomaly detection"
) -> str:
    """
    Generate AI-powered explanations for anomaly detection results.

    Uses PlotSenseAI's explanation engine to provide human-readable
    interpretations of anomaly detection results and visualizations.

    Parameters:
        df (pd.DataFrame): Anomaly detection results
        plot (plt.Figure, optional): The visualization to explain
        context (str, optional): Context for the explanation

    Returns:
        str: Human-readable explanation of the anomaly detection results

    Examples:
        >>> explanation = get_anomaly_explanation(results)
        >>> print(explanation)
        "The analysis detected 2 anomalies out of 12 data points (16.7% anomaly rate).
        The anomalous values (100, 120) are significantly higher than the normal
        range (10-15), suggesting potential outliers or measurement errors..."
    """
    if not PLOTSENSE_AVAILABLE:
        return _generate_basic_explanation(df)

    try:
        if plot is not None:
            # Use PlotSenseAI to explain the visualization
            explanation = explainer(plot)
            return explanation if explanation else _generate_basic_explanation(df)
        else:
            # Generate explanation based on data alone
            return _generate_basic_explanation(df)

    except Exception as e:
        warnings.warn(f"AI explanation failed: {e}", UserWarning)
        return _generate_basic_explanation(df)


def _enhance_anomaly_plot(plot: plt.Figure, df: pd.DataFrame) -> None:
    """
    Enhance the PlotSenseAI generated plot with anomaly-specific styling.

    Parameters:
        plot (plt.Figure): The plot to enhance
        df (pd.DataFrame): Anomaly detection results
    """
    try:
        # Get the main axes
        ax = plot.gca() if hasattr(plot, 'gca') else plot.axes[0]

        # Add title with anomaly statistics
        anomaly_count = df['anomaly'].sum()
        total_count = len(df)
        anomaly_rate = (anomaly_count / total_count) * 100

        title = f"Anomaly Detection Results: {anomaly_count}/{total_count} anomalies ({anomaly_rate:.1f}%)"
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add legend explaining colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Anomalies'),
            Patch(facecolor='blue', label='Normal Points')
        ]
        ax.legend(handles=legend_elements, loc='best')

    except Exception as e:
        # If enhancement fails, continue with basic plot
        warnings.warn(f"Plot enhancement failed: {e}", UserWarning)


def _fallback_visualization(df: pd.DataFrame, show_plot: bool = True) -> plt.Figure:
    """
    Create a basic anomaly visualization when PlotSense is not available.

    Parameters:
        df (pd.DataFrame): Anomaly detection results
        show_plot (bool): Whether to display the plot

    Returns:
        plt.Figure: Basic matplotlib plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Value vs Index with anomalies highlighted
    normal_data = df[~df['anomaly']]
    anomaly_data = df[df['anomaly']]

    ax1.scatter(normal_data.index, normal_data['value'],
                c='blue', alpha=0.6, label='Normal')
    ax1.scatter(anomaly_data.index, anomaly_data['value'],
                c='red', s=100, alpha=0.8, label='Anomaly')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Anomaly Detection Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Z-score distribution
    ax2.bar(range(len(df)), df['zscore'],
            color=['red' if x else 'blue' for x in df['anomaly']], alpha=0.7)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Z-Score')
    ax2.set_title('Z-Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig


def _generate_basic_explanation(df: pd.DataFrame) -> str:
    """
    Generate a basic explanation of anomaly detection results.

    Parameters:
        df (pd.DataFrame): Anomaly detection results

    Returns:
        str: Basic explanation string
    """
    anomaly_count = df['anomaly'].sum()
    total_count = len(df)
    anomaly_rate = (anomaly_count / total_count) * 100

    if anomaly_count == 0:
        return f"No anomalies detected in {total_count} data points. All values appear normal."

    anomaly_values = df[df['anomaly']]['value'].tolist()
    mean_val = df['value'].mean()
    std_val = df['value'].std()

    explanation = f"""
Anomaly Detection Summary:
- Total data points: {total_count}
- Anomalies detected: {anomaly_count}
- Anomaly rate: {anomaly_rate:.1f}%
- Anomalous values: {anomaly_values}
- Data mean: {mean_val:.2f}
- Data std deviation: {std_val:.2f}

The detected anomalies deviate significantly from the normal pattern in the dataset.
    """.strip()

    return explanation


def create_anomaly_dashboard(
    results_list: List[pd.DataFrame],
    labels: List[str],
    title: str = "Anomaly Detection Dashboard"
) -> plt.Figure:
    """
    Create a dashboard comparing multiple anomaly detection results.

    Parameters:
        results_list (List[pd.DataFrame]): List of anomaly detection results
        labels (List[str]): Labels for each result set
        title (str): Dashboard title

    Returns:
        plt.Figure: Dashboard figure with multiple subplots
    """
    n_results = len(results_list)
    fig, axes = plt.subplots(2, (n_results + 1) // 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    if n_results == 1:
        axes = [axes]
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i, (df, label) in enumerate(zip(results_list, labels)):
        row = i // ((n_results + 1) // 2)
        col = i % ((n_results + 1) // 2)
        ax = axes[row, col] if n_results > 1 else axes[i]

        # Plot anomalies
        normal_data = df[~df['anomaly']]
        anomaly_data = df[df['anomaly']]

        ax.scatter(normal_data.index, normal_data['value'],
                  c='blue', alpha=0.6, label='Normal')
        ax.scatter(anomaly_data.index, anomaly_data['value'],
                  c='red', s=60, alpha=0.8, label='Anomaly')

        anomaly_count = df['anomaly'].sum()
        ax.set_title(f'{label}\n({anomaly_count} anomalies)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    if n_results % 2 == 1 and n_results > 1:
        axes[-1, -1].set_visible(False)

    plt.tight_layout()
    return fig