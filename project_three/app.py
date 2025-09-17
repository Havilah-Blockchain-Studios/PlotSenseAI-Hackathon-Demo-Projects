"""
PlotSenseAI Data Storytelling Web Application

An interactive Streamlit web app that demonstrates the power of PlotSenseAI
for automated data visualization and explanation. This application allows
users to explore climate data through an intuitive interface and get
AI-powered insights.

Features:
- Interactive data exploration with filtering
- AI-powered visualization recommendations
- Real-time plot generation
- Automated explanations using Groq API
- Responsive design for different screen sizes

Usage:
    streamlit run app.py

Then navigate to http://localhost:8501 in your browser.

Requirements:
- streamlit
- pandas
- plotsense
- matplotlib (implicit via plotsense)

Optional:
- Groq API key for AI explanations

Authors: Havilah Academy Team for PlotSenseAI
License: MIT
"""

import streamlit as st
import pandas as pd
import os
import warnings
from typing import Optional, Dict, Any

# Import PlotSenseAI components
try:
    from plotsense import recommender, plotgen, explainer
    PLOTSENSE_AVAILABLE = True
except ImportError as e:
    PLOTSENSE_AVAILABLE = False
    st.error(f"PlotSense import failed: {e}")
    st.stop()

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and preprocess the climate dataset.

    This function loads the climate data from CSV, performs basic preprocessing,
    and caches the result for improved performance.

    Returns:
        pd.DataFrame: Preprocessed climate dataset with proper data types

    Raises:
        FileNotFoundError: If the climate.csv file is not found
        ValueError: If the data format is invalid
    """
    try:
        # Load the CSV file
        df = pd.read_csv("data/climate.csv")

        # Validate required columns
        required_columns = ["Date", "City", "Temperature", "Humidity", "Wind Speed", "Rainfall"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert Date column to datetime for better handling
        df["Date"] = pd.to_datetime(df["Date"])

        # Basic data validation
        if len(df) == 0:
            raise ValueError("Dataset is empty")

        # Sort by date for consistent display
        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except FileNotFoundError:
        st.error("❌ Climate data file not found. Please ensure 'data/climate.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract basic information about the dataset.

    Args:
        df (pd.DataFrame): The dataset to analyze

    Returns:
        Dict[str, Any]: Dictionary containing dataset statistics
    """
    return {
        "total_records": len(df),
        "date_range": {
            "start": df["Date"].min(),
            "end": df["Date"].max()
        },
        "cities": sorted(df["City"].unique().tolist()),
        "variables": ["Temperature", "Humidity", "Wind Speed", "Rainfall"],
        "missing_values": df.isnull().sum().sum()
    }


# Load the main dataset
df = load_data()
data_info = get_data_info(df)

# -------- Sidebar Controls --------
st.sidebar.title("Climate Data Storytelling")

# API key (hidden input)
groq_key = st.sidebar.text_input(
    "Enter your Groq API key",
    type="password",
    help="Get your key at https://console.groq.com/keys",
)
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key  # Set it for PlotSense

city = st.sidebar.selectbox("Select City", df["City"].unique())
variable = st.sidebar.selectbox(
    "Select Variable", ["Temperature", "Humidity", "Wind Speed", "Rainfall"]
)
n_recs = st.sidebar.slider("Number of PlotSense Suggestions", 1, 5, 3)

# Show raw data option
show_raw = st.sidebar.checkbox("Show raw data")

# -------- Filter Data --------
city_data = df[df["City"] == city]

# -------- Raw Data Preview --------
if show_raw:
    st.subheader(f"First 10 rows of {city} data")
    st.dataframe(city_data.head(10))

# -------- Recommendations --------
st.subheader(f"PlotSense Recommendations for {variable} in {city}")
suggestions = recommender(city_data[[variable, "Date"]], n=n_recs)
st.dataframe(suggestions)

# -------- Visualization --------
st.subheader("Visualization")
choice = st.selectbox("Choose a suggestion index", suggestions.index)
plot = plotgen(city_data, choice, suggestions)
st.pyplot(plot)

# -------- Explanation --------
st.subheader("Explanation")
if groq_key:
    explanation = explainer(plot)
    if explanation:
        st.write(str(explanation))  # ✅ always show text
    else:
        st.warning("No explanation returned. Try another visualization.")
else:
    st.info("Enter your Groq API key in the sidebar to see explanations.")
