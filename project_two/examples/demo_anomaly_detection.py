"""
PlotSenseAI Anomaly Detection Demo

This demo showcases the PlotSenseAI anomaly detection plugin in action.
It demonstrates how to:
1. Generate synthetic data with intentional anomalies
2. Detect anomalies using Z-score method
3. Visualize results with PlotSenseAI integration
4. Generate AI-powered explanations

Usage:
    python examples/demo_anomaly_detection.py

Expected Output:
- Console output showing detection statistics
- PlotSenseAI visualization highlighting anomalies
- AI-generated explanation of the results

Authors: Havilah Academy Team for PlotSenseAI
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our custom anomaly detection functions
from plotsense_anomaly.detection import (
    zscore_anomaly_detection,
    get_anomaly_summary,
    validate_threshold
)
from plotsense_anomaly.visualization import (
    visualize_anomalies,
    get_anomaly_explanation
)


def generate_sample_data(n_points: int = 100, anomaly_rate: float = 0.05) -> list:
    """
    Generate synthetic dataset with intentional anomalies for demonstration.

    Parameters:
        n_points (int): Total number of data points to generate
        anomaly_rate (float): Proportion of points that should be anomalies

    Returns:
        list: Generated data with anomalies mixed in
    """
    # Set random seed for reproducible results
    np.random.seed(42)

    # Generate normal data points (following normal distribution)
    normal_points = int(n_points * (1 - anomaly_rate))
    normal_data = np.random.normal(loc=50, scale=10, size=normal_points)

    # Generate anomalous data points (extreme values)
    anomaly_points = n_points - normal_points
    anomalies = []

    # Create different types of anomalies
    for i in range(anomaly_points):
        if i % 2 == 0:
            # High-value anomalies
            anomalies.append(np.random.uniform(120, 150))
        else:
            # Low-value anomalies
            anomalies.append(np.random.uniform(-10, 5))

    # Combine and shuffle
    all_data = list(normal_data) + anomalies
    np.random.shuffle(all_data)

    return all_data


def run_basic_demo():
    """Run the basic anomaly detection demonstration."""
    print("🔍 PlotSenseAI Anomaly Detection Demo")
    print("=" * 50)

    # Generate sample data
    print("📊 Generating synthetic dataset with intentional anomalies...")
    data = generate_sample_data(n_points=100, anomaly_rate=0.07)

    print(f"📈 Generated {len(data)} data points")
    print(f"📊 Data range: [{min(data):.2f}, {max(data):.2f}]")
    print(f"📊 Data statistics:")
    print(f"   Mean: {np.mean(data):.2f}")
    print(f"   Std Dev: {np.std(data):.2f}")

    # Validate threshold before using
    threshold = 2.0
    if validate_threshold(threshold):
        print(f"✅ Using threshold: {threshold}")
    else:
        print(f"⚠️ Invalid threshold: {threshold}, using default")
        threshold = 2.0

    print("\n🚨 Running anomaly detection...")

    # Perform anomaly detection
    results = zscore_anomaly_detection(data, threshold=threshold)

    # Get summary statistics
    summary = get_anomaly_summary(results)

    # Display results
    print("🎯 Anomaly Detection Results:")
    print(f"   Total data points: {summary['total_points']}")
    print(f"   Anomalies detected: {summary['anomaly_count']}")
    print(f"   Anomaly rate: {summary['anomaly_rate']:.1%}")
    print(f"   Detection threshold: {summary['threshold']}")
    print(f"   Maximum Z-score: {summary['max_zscore']:.2f}")

    if summary['anomaly_values']:
        print("\n🎯 Anomalous values:")
        for i, (value, zscore) in enumerate(zip(summary['anomaly_values'], summary['extreme_zscores'])):
            print(f"   {i+1}. Value: {value:.2f} (Z-score: {zscore:.2f})")

    print("\n📊 Generating PlotSenseAI visualization...")

    try:
        # Create visualization
        plot, recommendations = visualize_anomalies(
            results,
            n=3,
            return_recommendations=True,
            show_plot=False  # We'll show it manually after adding explanations
        )

        print("✅ Visualization generated successfully!")
        print("\n📋 PlotSenseAI Recommendations:")
        print(recommendations.to_string(index=False))

        # Generate explanation
        print("\n🧠 Generating AI explanation...")
        explanation = get_anomaly_explanation(results, plot)
        print("\n💡 AI Explanation:")
        print("-" * 30)
        print(explanation)
        print("-" * 30)

        # Display the plot
        print("\n🎨 Displaying visualization...")
        plot.show()

        return results, plot, explanation

    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        print("📊 Showing basic statistics instead")
        return results, None, None


def run_interactive_demo():
    """Run an interactive demo allowing user to adjust parameters."""
    print("\n🎮 Interactive Anomaly Detection Demo")
    print("=" * 50)

    try:
        # Get user input for parameters
        print("\n⚙️ Configure Detection Parameters:")

        # Data size
        while True:
            try:
                n_points = int(input("Number of data points (50-500, default 100): ") or "100")
                if 50 <= n_points <= 500:
                    break
                else:
                    print("Please enter a number between 50 and 500")
            except ValueError:
                print("Please enter a valid number")

        # Threshold
        while True:
            try:
                threshold = float(input("Z-score threshold (1.0-4.0, default 2.0): ") or "2.0")
                if validate_threshold(threshold):
                    break
                else:
                    print("Please enter a valid threshold between 1.0 and 4.0")
            except ValueError:
                print("Please enter a valid number")

        # Generate and analyze data
        data = generate_sample_data(n_points=n_points, anomaly_rate=0.05)
        results = zscore_anomaly_detection(data, threshold=threshold)
        summary = get_anomaly_summary(results)

        print(f"\n📊 Results with your parameters:")
        print(f"   Data points: {n_points}")
        print(f"   Threshold: {threshold}")
        print(f"   Anomalies found: {summary['anomaly_count']} ({summary['anomaly_rate']:.1%})")

        # Ask about visualization
        show_viz = input("\nShow visualization? (y/n, default y): ").lower()
        if show_viz != 'n':
            plot = visualize_anomalies(results, n=2)
            explanation = get_anomaly_explanation(results, plot)
            print(f"\n💡 Explanation: {explanation}")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")


def compare_thresholds_demo():
    """Demonstrate the effect of different threshold values."""
    print("\n🔬 Threshold Comparison Demo")
    print("=" * 50)

    # Generate consistent dataset
    data = generate_sample_data(n_points=80, anomaly_rate=0.1)
    thresholds = [1.5, 2.0, 2.5, 3.0]

    results_list = []
    labels = []

    print("📊 Comparing different threshold values...")

    for threshold in thresholds:
        print(f"\n⚙️ Testing threshold: {threshold}")

        # Run detection
        results = zscore_anomaly_detection(data, threshold=threshold)
        summary = get_anomaly_summary(results)

        print(f"   Anomalies detected: {summary['anomaly_count']}")
        print(f"   Anomaly rate: {summary['anomaly_rate']:.1%}")

        results_list.append(results)
        labels.append(f"Threshold: {threshold}")

    # Create comparison dashboard
    try:
        from plotsense_anomaly.visualization import create_anomaly_dashboard

        print("\n📊 Creating comparison dashboard...")
        dashboard = create_anomaly_dashboard(
            results_list,
            labels,
            title="Threshold Comparison: Effect on Anomaly Detection"
        )
        dashboard.show()

        print("✅ Dashboard created successfully!")
        print("\n💡 Observation: Lower thresholds detect more anomalies")
        print("   but may include false positives. Higher thresholds are")
        print("   more conservative but may miss subtle anomalies.")

    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")


def main():
    """Main function to run all demo scenarios."""
    print("🚀 Welcome to PlotSenseAI Anomaly Detection Plugin Demo!")
    print("🕐 Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    try:
        # Run basic demo
        basic_results = run_basic_demo()

        # Ask user if they want to continue with other demos
        print("\n" + "=" * 60)
        continue_demo = input("Continue with interactive demo? (y/n, default n): ").lower()

        if continue_demo == 'y':
            run_interactive_demo()

            print("\n" + "=" * 60)
            threshold_demo = input("Show threshold comparison demo? (y/n, default n): ").lower()

            if threshold_demo == 'y':
                compare_thresholds_demo()

        print("\n🎉 Demo completed successfully!")
        print("📖 Check out the documentation for more advanced features")
        print("🔗 GitHub: https://github.com/HavilahAcademy/PlotSenseAI-Hackathon-Demo-Projects")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        print("💡 Make sure PlotSense is properly installed")

    finally:
        print(f"\n🕐 Demo ended at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # This code runs when the script is executed directly
    main()