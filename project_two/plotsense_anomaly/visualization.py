# plotsense_anomaly/visualization.py

import plotsense as ps
from plotsense import plotgen, recommender

def visualize_anomalies(df, n=1):
    """
    Use PlotSense to visualize anomalies in the dataset.
    
    Parameters:
        df (pd.DataFrame): Dataframe with anomaly column included.
        n (int): Number of recommended visualizations to generate.
    """
    # Ask PlotSense for visualization suggestions
    suggestions = recommender(df, n=n)

    # Generate first recommended plot
    plot = plotgen(df, 0, suggestions)
    return plot
