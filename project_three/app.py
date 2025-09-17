import streamlit as st
import pandas as pd
import os

from plotsense import recommender, plotgen, explainer

# -------- Load Data --------
@st.cache_data
def load_data():
    df = pd.read_csv("data/climate.csv")
    # Make sure Date is treated as datetime
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

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
