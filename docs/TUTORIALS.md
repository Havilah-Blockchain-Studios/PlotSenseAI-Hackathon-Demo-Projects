# PlotSenseAI Tutorials 📚

Welcome to the comprehensive tutorial series for PlotSenseAI! These tutorials will guide you through each demo project step-by-step, helping you understand both the code and the concepts behind AI-powered data visualization.

## 🎯 Learning Objectives

By completing these tutorials, you will:
- Master PlotSenseAI's core functionality
- Understand ML explainability techniques
- Learn to build custom data visualization plugins
- Create interactive web applications for data storytelling
- Apply best practices for AI-driven data analysis

## 📖 Tutorial Structure

Each tutorial includes:
- **🎯 Objectives**: What you'll learn
- **⚙️ Prerequisites**: Required knowledge and setup
- **👣 Step-by-step guide**: Detailed instructions
- **🧪 Exercises**: Hands-on practice
- **🔧 Troubleshooting**: Common issues and solutions
- **🚀 Next steps**: How to extend and improve

---

## 🔍 Tutorial 1: ML Explainability with PlotSenseAI

### 🎯 Objectives
- Load and preprocess real-world datasets
- Train machine learning models
- Use PlotSenseAI for automated visualization recommendations
- Generate and interpret AI explanations
- Explore advanced explainability techniques

### ⚙️ Prerequisites
- Basic Python knowledge
- Understanding of pandas and scikit-learn
- Jupyter Notebook setup (see [SETUP.md](./SETUP.md))

### 👣 Step-by-Step Guide

#### Step 1: Environment Setup
```bash
cd project_one
pip install ucimlrepo scikit-learn pandas matplotlib plotsense
jupyter notebook ml_explainability_demo.ipynb
```

#### Step 2: Understanding the Dataset
The UCI Breast Cancer Recurrence dataset contains:
- **Features**: Age, menopause status, tumor size, etc.
- **Target**: Recurrence (no-recurrence-events vs recurrence-events)
- **Challenge**: Imbalanced classes and categorical features

```python
# Load the dataset
from ucimlrepo import fetch_ucirepo
breast_cancer_recurrence = fetch_ucirepo(id=14)
X = breast_cancer_recurrence.data.features
y = breast_cancer_recurrence.data.targets
```

#### Step 3: Data Preprocessing
```python
# Handle missing values
X_cleaned = X.fillna(X.mode().iloc[0])

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in X_cleaned.select_dtypes(include=['object']).columns:
    X_cleaned[col] = le.fit_transform(X_cleaned[col])
```

#### Step 4: Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

#### Step 5: PlotSenseAI Integration
```python
from plotsense import recommender, plotgen, explainer

# Get visualization recommendations
recommendations = recommender(X_train, n=5)
print("PlotSenseAI Recommendations:")
print(recommendations)

# Generate visualization
plot = plotgen(X_train, 0, recommendations)  # Use first recommendation
plot.show()

# Get AI explanation
explanation = explainer(plot)
print("AI Explanation:", explanation)
```

#### Step 6: Model Explainability
```python
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize with PlotSenseAI
importance_plot_recs = recommender(feature_importance, n=3)
importance_plot = plotgen(feature_importance, 0, importance_plot_recs)
importance_plot.show()
```

### 🧪 Exercises

1. **Data Exploration**:
   - Try different datasets from UCI repository
   - Experiment with various preprocessing techniques
   - Compare PlotSenseAI recommendations for different data types

2. **Model Comparison**:
   - Train different models (SVM, Logistic Regression, XGBoost)
   - Use PlotSenseAI to visualize model performance comparisons
   - Generate explanations for each model's behavior

3. **Advanced Explainability**:
   - Implement SHAP values visualization
   - Create partial dependence plots
   - Explore feature interaction effects

### 🔧 Troubleshooting

**Issue**: UCI dataset not loading
```python
# Alternative: Use built-in datasets
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
```

**Issue**: PlotSenseAI recommendations seem irrelevant
- Check data types and ensure proper preprocessing
- Try different subsets of your data
- Experiment with different `n` values for recommendations

---

## 🚨 Tutorial 2: Building Custom Anomaly Detection Plugin

### 🎯 Objectives
- Create modular Python packages
- Implement statistical anomaly detection
- Integrate custom functionality with PlotSenseAI
- Write comprehensive unit tests
- Package and distribute Python modules

### ⚙️ Prerequisites
- Python packaging knowledge
- Understanding of statistical concepts (Z-score, standard deviation)
- Basic testing with pytest

### 👣 Step-by-Step Guide

#### Step 1: Project Structure Setup
```bash
cd project_two
ls -la  # Observe the package structure
```

The structure follows Python packaging best practices:
```
plotsense_anomaly/
├── __init__.py          # Package initialization
├── detection.py         # Core anomaly detection logic
└── visualization.py     # PlotSenseAI integration
```

#### Step 2: Understanding the Detection Algorithm
```python
# File: plotsense_anomaly/detection.py
def zscore_anomaly_detection(data, threshold=1.5):
    """
    Z-score based anomaly detection

    Anomaly if: |Z-score| > threshold
    Z-score = (value - mean) / standard_deviation
    """
    df = pd.DataFrame({"value": data})
    mean = df["value"].mean()
    std = df["value"].std()

    df["zscore"] = (df["value"] - mean) / std
    df["anomaly"] = np.abs(df["zscore"]) > threshold
    return df
```

#### Step 3: PlotSenseAI Integration
```python
# File: plotsense_anomaly/visualization.py
from plotsense import recommender, plotgen

def visualize_anomalies(data, anomalies):
    # Create visualization dataset
    viz_data = pd.DataFrame({
        'value': data,
        'anomaly': anomalies,
        'index': range(len(data))
    })

    # Get PlotSenseAI recommendations
    recommendations = recommender(viz_data, n=3)

    # Generate plot
    plot = plotgen(viz_data, 0, recommendations)
    return plot
```

#### Step 4: Running the Demo
```bash
python examples/demo_anomaly_detection.py
```

Expected output:
```
Detected 3 anomalies out of 100 data points
Anomalous values: [45.2, -32.1, 67.8]
[PlotSenseAI visualization appears]
```

#### Step 5: Testing
```bash
python -m pytest tests/test_detection.py -v
```

### 🧪 Exercises

1. **Algorithm Enhancement**:
   - Implement IQR-based anomaly detection
   - Add support for multivariate anomaly detection
   - Create ensemble methods combining multiple techniques

2. **Visualization Improvements**:
   - Add color coding for different anomaly types
   - Create interactive hover information
   - Implement time-series specific visualizations

3. **Package Extension**:
   - Add configuration files for different detection parameters
   - Create CLI interface for the package
   - Add support for streaming data

### 🔧 Troubleshooting

**Issue**: Import errors when running examples
```bash
# Install package in development mode
pip install -e .
```

**Issue**: Tests failing
- Check that all dependencies are installed
- Verify Python path includes the project directory
- Run tests with more verbose output: `pytest -v -s`

---

## 📊 Tutorial 3: Interactive Data Storytelling Web App

### 🎯 Objectives
- Build responsive web applications with Streamlit
- Create interactive data exploration interfaces
- Integrate multiple PlotSenseAI features
- Handle user input and API key management
- Deploy data applications

### ⚙️ Prerequisites
- Basic web development concepts
- Streamlit framework basics
- Understanding of API integration

### 👣 Step-by-Step Guide

#### Step 1: Application Architecture
```python
# File: app.py - Key components

# Data Loading (with caching)
@st.cache_data
def load_data():
    return pd.read_csv("data/climate.csv")

# Sidebar Controls
city = st.sidebar.selectbox("Select City", df["City"].unique())
variable = st.sidebar.selectbox("Select Variable", ["Temperature", "Humidity"])

# Main Content
recommendations = recommender(filtered_data, n=3)
plot = plotgen(filtered_data, choice, recommendations)
explanation = explainer(plot)
```

#### Step 2: Running the Application
```bash
cd project_three
pip install -r requirements.txt
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

#### Step 3: Understanding the User Interface

**Sidebar Components**:
- API Key Input (hidden/password type)
- City Selection (dropdown)
- Variable Selection (dropdown)
- Number of recommendations (slider)
- Raw data toggle (checkbox)

**Main Content**:
- Data preview table
- PlotSenseAI recommendations table
- Interactive visualization
- AI-generated explanations

#### Step 4: Data Flow
1. User selects parameters in sidebar
2. Data gets filtered based on selections
3. PlotSenseAI generates recommendations
4. User chooses a recommendation
5. Visualization is generated and displayed
6. AI explanation is generated (if API key provided)

#### Step 5: Customization Examples

**Adding New Variables**:
```python
# In the sidebar section
new_variable = st.sidebar.selectbox(
    "Select New Variable",
    ["Temperature", "Humidity", "Wind Speed", "Rainfall", "Pressure"]
)
```

**Custom Filtering**:
```python
# Add date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[df["Date"].min(), df["Date"].max()],
    min_value=df["Date"].min(),
    max_value=df["Date"].max()
)

# Filter data
filtered_data = df[
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]
```

### 🧪 Exercises

1. **UI Enhancement**:
   - Add multiple city selection
   - Implement data export functionality
   - Create comparison views between cities

2. **Advanced Features**:
   - Add real-time data updates
   - Implement user authentication
   - Create dashboard with multiple charts

3. **Deployment**:
   - Deploy to Streamlit Cloud
   - Create Docker container
   - Set up environment variables for production

### 🔧 Troubleshooting

**Issue**: App not loading data
- Check that `data/climate.csv` exists
- Verify file path in `load_data()` function
- Ensure data file has expected columns

**Issue**: PlotSenseAI not working
- Verify plotsense installation: `pip show plotsense`
- Check for API key requirements
- Test PlotSenseAI in isolation first

---

## 🚀 Advanced Topics

### Combining Multiple Projects

**Project Integration Example**:
```python
# Combine anomaly detection with web app
from plotsense_anomaly import zscore_anomaly_detection
from plotsense import recommender, plotgen

# In your Streamlit app
anomalies = zscore_anomaly_detection(data, threshold=2.0)
anomaly_recs = recommender(anomalies, n=5)
anomaly_plot = plotgen(anomalies, 0, anomaly_recs)
st.pyplot(anomaly_plot)
```

### Performance Optimization

**Data Caching**:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_computation(data):
    return processed_data
```

**Lazy Loading**:
```python
if st.button("Generate Advanced Analysis"):
    with st.spinner("Computing..."):
        result = complex_analysis(data)
        st.success("Analysis complete!")
```

### Production Considerations

1. **Environment Variables**: Use for API keys and configuration
2. **Error Handling**: Implement comprehensive try-catch blocks
3. **Logging**: Add logging for debugging and monitoring
4. **Testing**: Create integration tests for web components
5. **Security**: Validate user inputs and sanitize data

## 🎓 Next Steps

After completing these tutorials:

1. **Contribute**: Submit improvements to the demo projects
2. **Create**: Build your own PlotSenseAI applications
3. **Share**: Present your work at the hackathon
4. **Learn**: Explore advanced PlotSenseAI features
5. **Connect**: Join the PlotSenseAI community

## 📚 Additional Resources

- [PlotSenseAI API Documentation](https://docs.plotsense.ai)
- [Streamlit Tutorials](https://docs.streamlit.io/get-started/tutorials)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Packaging Guide](https://packaging.python.org/tutorials/)

---

**Happy Learning! 🎉**