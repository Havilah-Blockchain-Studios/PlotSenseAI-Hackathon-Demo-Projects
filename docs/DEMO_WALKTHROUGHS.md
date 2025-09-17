# Demo Walkthroughs 🎮

Welcome to the comprehensive demo walkthroughs! These guides will take you step-by-step through each demo project, showing you exactly what to expect and how to interact with the applications.

## 🎯 Walkthrough Structure

Each walkthrough includes:
- **📖 Overview**: What the demo does
- **⚡ Quick Start**: Get running in 2 minutes
- **🎮 Interactive Guide**: Step-by-step usage
- **🎯 Key Features**: What to focus on
- **🧪 Experiments**: Things to try
- **🔧 Customization**: How to modify and extend

---

# 🔍 Demo 1: ML Explainability Walkthrough

## 📖 Overview
This Jupyter notebook demonstrates how PlotSenseAI can make machine learning models more interpretable by automatically generating visualizations and explanations for model predictions.

## ⚡ Quick Start (2 minutes)

```bash
cd project_one
pip install ucimlrepo scikit-learn pandas matplotlib plotsense
jupyter notebook ml_explainability_demo.ipynb
```

Open your browser to `http://localhost:8888` and click on `ml_explainability_demo.ipynb`.

## 🎮 Interactive Walkthrough

### Step 1: Understanding the Data 📊
When you run the first few cells, you'll see:

```python
# Cell 1-2: Data Loading
from ucimlrepo import fetch_ucirepo
breast_cancer_recurrence = fetch_ucirepo(id=14)
```

**What happens**: Downloads the UCI Breast Cancer Recurrence dataset
**Look for**:
- Dataset shape and size
- Feature names and types
- Missing value patterns

### Step 2: Data Exploration 🔍
```python
# Cell 3-4: Initial Exploration
print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target distribution: {y.value_counts()}")
```

**Expected Output**:
```
Dataset shape: (286, 9)
Features: ['age', 'menopause', 'tumor-size', 'inv-nodes', ...]
Target distribution:
no-recurrence-events    201
recurrence-events        85
```

**Key Insight**: Notice the class imbalance - this is a real-world challenge!

### Step 3: Data Preprocessing 🛠️
```python
# Cell 5-6: Cleaning and Encoding
X_cleaned = X.fillna(X.mode().iloc[0])
for col in X_cleaned.select_dtypes(include=['object']).columns:
    X_cleaned[col] = le.fit_transform(X_cleaned[col])
```

**Watch for**:
- How categorical variables get encoded
- Missing value handling strategy
- Data type transformations

### Step 4: PlotSenseAI Magic ✨
```python
# Cell 7-8: First PlotSenseAI Usage
from plotsense import recommender, plotgen, explainer

recommendations = recommender(X_cleaned, n=5)
print("PlotSenseAI Recommendations:")
display(recommendations)
```

**Expected Output**:
```
   Recommendation                     Confidence  Chart_Type
0  Feature correlation heatmap            0.92    heatmap
1  Distribution comparison               0.87    boxplot
2  Feature importance ranking            0.83    barplot
3  Scatter plot matrix                   0.78    scatter
4  Violin plot comparison                0.71    violin
```

**💡 Key Point**: PlotSenseAI automatically analyzes your data and suggests the most relevant visualizations!

### Step 5: Model Training 🤖
```python
# Cell 9-10: ML Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.3f}")
```

**Expected Output**: `Model accuracy: 0.754`

### Step 6: Visualization Generation 📈
```python
# Cell 11-12: Generate Plot
plot = plotgen(X_train, 0, recommendations)  # Use first recommendation
plot.show()
```

**What you'll see**: An automatically generated heatmap showing feature correlations, with:
- Professional styling and color schemes
- Proper axis labels and titles
- Clear correlation patterns highlighted

### Step 7: AI Explanations 🧠
```python
# Cell 13: Get Explanation
explanation = explainer(plot)
print("AI Explanation:")
print(explanation)
```

**Sample Output**:
```
"This correlation heatmap reveals important relationships in the breast cancer dataset.
Strong positive correlations appear between tumor-size and inv-nodes (0.67), suggesting
larger tumors are associated with more invasive nodes. The age feature shows weak
correlations with other variables, indicating it may be less predictive..."
```

### Step 8: Feature Importance Analysis 🎯
```python
# Cell 14-15: Model Explainability
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Use PlotSenseAI for feature importance visualization
importance_recs = recommender(feature_importance, n=3)
importance_plot = plotgen(feature_importance, 0, importance_recs)
importance_plot.show()
```

**What you'll discover**:
- Which features most influence model predictions
- How PlotSenseAI adapts recommendations to different data types
- Clear visual hierarchy of feature importance

## 🎯 Key Features to Explore

### 1. Recommendation Adaptation
Try changing the data subset:
```python
# Try with different feature subsets
numeric_only = X_train.select_dtypes(include=[np.number])
categorical_only = X_train.select_dtypes(include=['object'])

recs_numeric = recommender(numeric_only, n=3)
recs_categorical = recommender(categorical_only, n=3)
```

**Observation**: Notice how recommendations change based on data types!

### 2. Interactive Exploration
```python
# Try different recommendation indices
for i in range(len(recommendations)):
    print(f"\\n--- Visualization {i+1}: {recommendations.iloc[i]['Recommendation']} ---")
    plot = plotgen(X_train, i, recommendations)
    plot.show()

    explanation = explainer(plot)
    print(f"Explanation: {explanation}")
```

### 3. Model Comparison
```python
# Compare different models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: {score:.3f}")
```

## 🧪 Experiments to Try

1. **Different Datasets**: Replace with other UCI datasets
2. **Feature Engineering**: Create new features and see how recommendations change
3. **Model Types**: Try deep learning models and compare explanations
4. **Custom Thresholds**: Experiment with different confidence thresholds for recommendations

---

# 🚨 Demo 2: Anomaly Detection Plugin Walkthrough

## 📖 Overview
This demo shows how to extend PlotSenseAI with custom functionality by building an anomaly detection plugin that integrates seamlessly with PlotSenseAI's visualization engine.

## ⚡ Quick Start (2 minutes)

```bash
cd project_two
pip install -r requirements.txt
python examples/demo_anomaly_detection.py
```

## 🎮 Interactive Walkthrough

### Step 1: Understanding the Plugin Architecture 🏗️

**File Structure Overview**:
```
plotsense_anomaly/
├── __init__.py          # Package initialization
├── detection.py         # Core anomaly detection logic
└── visualization.py     # PlotSenseAI integration
```

**Key Concept**: Modular design allows easy extension and testing

### Step 2: Exploring the Detection Algorithm 🔍

Open `plotsense_anomaly/detection.py`:

```python
def zscore_anomaly_detection(data, threshold=1.5):
    """
    Z-score based anomaly detection
    Anomaly if: |Z-score| > threshold
    """
    df = pd.DataFrame({"value": data})
    mean = df["value"].mean()
    std = df["value"].std()

    df["zscore"] = (df["value"] - mean) / std
    df["anomaly"] = np.abs(df["zscore"]) > threshold
    return df
```

**Understanding Z-score**:
- Measures how many standard deviations away from the mean
- threshold=1.5 means values 1.5+ std devs away are anomalies
- Common thresholds: 1.5 (moderate), 2.0 (standard), 3.0 (conservative)

### Step 3: Running the Demo 🎬

```bash
python examples/demo_anomaly_detection.py
```

**Expected Output**:
```
🔍 PlotSense Anomaly Detection Demo

📊 Generated 100 data points with intentional anomalies
📈 Data range: [-2.45, 45.23]
📊 Data statistics:
   Mean: 10.23
   Std Dev: 8.67

🚨 Anomaly Detection Results:
   Total anomalies detected: 7
   Anomaly rate: 7.0%

🎯 Anomalous values:
   Index 23: 45.23 (Z-score: 4.04)
   Index 67: -2.45 (Z-score: -1.47)
   Index 89: 38.91 (Z-score: 3.31)
   ...

📊 Generating PlotSenseAI visualization...
```

**What happens next**: A visualization window opens showing:
- Scatter plot of all data points
- Anomalies highlighted in red
- Normal points in blue
- Clear threshold boundaries

### Step 4: Understanding the Visualization Integration 🎨

Open `plotsense_anomaly/visualization.py`:

```python
def visualize_anomalies(data, anomalies):
    viz_data = pd.DataFrame({
        'value': data,
        'anomaly': anomalies,
        'index': range(len(data))
    })

    recommendations = recommender(viz_data, n=3)
    plot = plotgen(viz_data, 0, recommendations)
    return plot
```

**Key Integration Points**:
1. **Data Preparation**: Structures data for PlotSenseAI
2. **Recommendation**: Gets visualization suggestions
3. **Generation**: Creates the actual plot

### Step 5: Testing the Plugin 🧪

```bash
python -m pytest tests/test_detection.py -v
```

**Expected Output**:
```
tests/test_detection.py::test_zscore_basic ✓
tests/test_detection.py::test_zscore_threshold ✓
tests/test_detection.py::test_zscore_edge_cases ✓
tests/test_detection.py::test_zscore_empty_data ✓

====== 4 passed in 0.23s ======
```

**What's being tested**:
- Basic functionality with normal data
- Different threshold values
- Edge cases (single value, all same values)
- Error handling (empty data)

## 🎯 Key Features to Explore

### 1. Threshold Sensitivity Analysis

```python
import numpy as np
from plotsense_anomaly import zscore_anomaly_detection

# Generate test data
np.random.seed(42)
data = np.random.normal(0, 1, 100)
data = np.append(data, [5, -5, 6])  # Add obvious anomalies

# Test different thresholds
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
for threshold in thresholds:
    result = zscore_anomaly_detection(data, threshold)
    anomaly_count = result['anomaly'].sum()
    print(f"Threshold {threshold}: {anomaly_count} anomalies")
```

**Expected Pattern**: Higher thresholds → fewer anomalies detected

### 2. Real-time Anomaly Detection

```python
# Simulate streaming data
import time
import matplotlib.pyplot as plt

def streaming_anomaly_demo():
    data_stream = []

    for i in range(50):
        # Normal data with occasional anomalies
        if i % 10 == 0:
            new_point = np.random.normal(0, 1) * 5  # Anomaly
        else:
            new_point = np.random.normal(0, 1)      # Normal

        data_stream.append(new_point)

        if len(data_stream) >= 10:  # Need minimum data for stats
            result = zscore_anomaly_detection(data_stream, threshold=2.0)
            latest_anomaly = result.iloc[-1]['anomaly']

            if latest_anomaly:
                print(f"🚨 ANOMALY at step {i}: {new_point:.2f}")
            else:
                print(f"✅ Normal at step {i}: {new_point:.2f}")

        time.sleep(0.1)  # Simulate real-time delay

streaming_anomaly_demo()
```

### 3. Multi-dimensional Anomaly Detection

```python
# Extend to 2D data
def multivariate_zscore(data_2d, threshold=2.0):
    """
    2D anomaly detection using Mahalanobis distance
    """
    import scipy.spatial.distance as distance

    # Calculate Mahalanobis distance for each point
    mean = np.mean(data_2d, axis=0)
    cov = np.cov(data_2d.T)

    distances = []
    for point in data_2d:
        dist = distance.mahalanobis(point, mean, np.linalg.inv(cov))
        distances.append(dist)

    distances = np.array(distances)
    threshold_val = np.percentile(distances, 95)  # Top 5% as anomalies

    return distances > threshold_val

# Test with 2D data
data_2d = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
anomalies_2d = multivariate_zscore(data_2d)
print(f"2D anomalies detected: {np.sum(anomalies_2d)}")
```

## 🧪 Experiments to Try

1. **Algorithm Comparison**: Implement IQR-based detection and compare
2. **Parameter Tuning**: Find optimal thresholds for different data types
3. **Integration Testing**: Use with real datasets from other projects
4. **Performance Testing**: Benchmark with large datasets

---

# 📊 Demo 3: Data Storytelling Web App Walkthrough

## 📖 Overview
An interactive Streamlit web application that demonstrates how PlotSenseAI can be integrated into web applications for intuitive data exploration and storytelling.

## ⚡ Quick Start (2 minutes)

```bash
cd project_three
pip install -r requirements.txt
streamlit run app.py
```

Browser opens to `http://localhost:8501`

## 🎮 Interactive Walkthrough

### Step 1: First Look at the Interface 👀

**Main Components**:
- **Sidebar**: Controls and configuration
- **Main Area**: Data display and visualizations
- **Status Bar**: Real-time feedback

**Initial State**: App loads with Chicago temperature data displayed

### Step 2: Understanding the Data 📊

The climate dataset contains:
- **Cities**: Chicago, New York, Phoenix, Los Angeles
- **Variables**: Temperature, Humidity, Wind Speed, Rainfall
- **Time Range**: Full year of daily data (2023)
- **Format**: Clean, structured CSV data

**Quick Exercise**: Check "Show raw data" in sidebar to see the data structure.

### Step 3: Basic Interaction Flow 🔄

#### A. Select Different City
1. **Current**: Chicago (default)
2. **Change to**: New York
3. **Observe**: Data updates automatically
4. **Notice**: PlotSenseAI recommendations change based on new data patterns

#### B. Change Variable
1. **Current**: Temperature
2. **Change to**: Humidity
3. **Observe**: Recommendations adapt to different data distribution
4. **Key Insight**: Different variables → different optimal visualizations

#### C. Adjust Recommendations
1. **Current**: 3 suggestions
2. **Change slider**: 5 suggestions
3. **Observe**: More visualization options appear
4. **Try**: Different recommendation indices

### Step 4: PlotSenseAI in Action ✨

#### Recommendation Analysis
When you change cities, watch the recommendations table:

**Chicago Temperature** might show:
```
Index  Recommendation              Confidence  Chart_Type
0      Time series line plot          0.94      line
1      Distribution histogram         0.87      histogram
2      Seasonal decomposition         0.82      seasonal
```

**Phoenix Humidity** might show:
```
Index  Recommendation              Confidence  Chart_Type
0      Box plot by month             0.91      boxplot
1      Scatter vs temperature        0.85      scatter
2      Violin plot seasonal          0.79      violin
```

**Key Observation**: PlotSenseAI adapts recommendations to:
- Data distribution characteristics
- Variable types and ranges
- Temporal patterns
- Correlation structures

#### Visualization Generation
1. **Select recommendation**: Choose index from dropdown
2. **Auto-generation**: Plot appears instantly
3. **Professional quality**: Clean styling, proper labels
4. **Interactive elements**: Hover, zoom, pan (depending on plot type)

### Step 5: AI Explanations 🧠

#### Setting Up API Key
1. **Get key**: Visit [Groq Console](https://console.groq.com/keys)
2. **Format**: `gsk_xxxxxxxxxxxxxxxxxxxxx`
3. **Enter**: In sidebar password field
4. **Test**: Generate a visualization

#### Understanding Explanations
Sample explanation for a temperature time series:

```
"This time series visualization of Chicago temperature data reveals clear seasonal patterns typical of continental climate zones. The data shows:

🌡️ Temperature Range: 15-85°F across the year
📈 Seasonal Trends: Clear winter lows (Jan-Feb) and summer highs (Jul-Aug)
📊 Variability: Higher day-to-day variation in spring/fall transition periods
🎯 Key Insights: The data suggests typical Midwest weather patterns with distinct seasonal cycles"
```

**What to look for**:
- **Data interpretation**: What the numbers mean
- **Pattern recognition**: Trends and anomalies identified
- **Context**: Real-world implications
- **Actionable insights**: What the patterns suggest

### Step 6: Advanced Exploration 🔍

#### Multi-City Comparison Exercise
1. **Start**: Chicago, Temperature
2. **Note**: Visualization characteristics
3. **Switch**: Phoenix, Temperature
4. **Compare**: How do patterns differ?
5. **Insight**: Desert vs. continental climate patterns

#### Cross-Variable Analysis
1. **Setup**: Same city, different variables
2. **Example**: Los Angeles
   - Temperature: Mild variations
   - Humidity: Inverse correlation with temperature
   - Rainfall: Sparse, seasonal clusters
   - Wind Speed: Consistent patterns

#### Time-based Patterns
1. **Observation**: Look for seasonal trends
2. **Comparison**: Compare similar months across variables
3. **Correlation**: Notice relationships between variables

## 🎯 Key Features to Explore

### 1. Responsive Design
- **Desktop**: Full sidebar layout
- **Mobile**: Collapsible sidebar
- **Tablet**: Optimized spacing

**Test**: Resize browser window to see adaptive layout

### 2. Real-time Updates
- **Data filtering**: Instant response to city changes
- **Visualization refresh**: Automatic plot updates
- **Recommendation adaptation**: Dynamic suggestion updates

### 3. Error Handling
**Try these edge cases**:
- Empty API key → Graceful degradation
- Network issues → Appropriate error messages
- Invalid selections → Auto-correction

### 4. Performance Optimization
**Notice**:
- **Caching**: Data loads only once (`@st.cache_data`)
- **Lazy loading**: Explanations only when API key provided
- **Efficient updates**: Only changed components re-render

## 🧪 Experiments to Try

### 1. Data Customization
Replace `climate.csv` with your own dataset:
```python
# Required columns: Date, Category, Numeric_Variable
# Example: sales.csv with Date, Region, Revenue
```

### 2. Feature Extensions
Add new sidebar controls:
```python
# Date range picker
date_range = st.sidebar.date_input("Select Date Range")

# Multiple city selection
cities = st.sidebar.multiselect("Select Cities", df["City"].unique())

# Custom thresholds
threshold = st.sidebar.slider("Anomaly Threshold", 1.0, 3.0, 2.0)
```

### 3. Visualization Enhancements
```python
# Add plot customization options
plot_style = st.sidebar.selectbox("Plot Style", ["default", "dark", "minimal"])
color_scheme = st.sidebar.color_picker("Choose Color")
```

### 4. Integration with Other Demos
```python
# Combine with anomaly detection
from plotsense_anomaly import zscore_anomaly_detection

# Add anomaly detection toggle
if st.sidebar.checkbox("Detect Anomalies"):
    anomalies = zscore_anomaly_detection(filtered_data[variable])
    st.subheader("Anomaly Detection Results")
    st.write(f"Anomalies detected: {anomalies['anomaly'].sum()}")
```

## 🔧 Customization Guide

### Adding New Variables
1. **Update data**: Add columns to CSV
2. **Update UI**: Add to selectbox options
3. **Test**: Verify PlotSenseAI handles new data types

### Custom Styling
```python
# Add custom CSS
st.markdown("""
<style>
.main-header {
    color: #1f77b4;
    font-size: 2rem;
}
</style>
""", unsafe_allow_html=True)
```

### New Visualization Types
```python
# Custom plot function
def custom_plot_type(data, variable):
    # Your custom visualization logic
    fig, ax = plt.subplots()
    # ... plotting code ...
    return fig
```

---

# 🎯 Summary and Next Steps

## 🏆 What You've Accomplished

After completing these walkthroughs, you've:

✅ **Mastered PlotSenseAI Basics**: Recommendations, generation, explanations
✅ **Built Custom Extensions**: Created anomaly detection plugin
✅ **Developed Web Applications**: Interactive data storytelling app
✅ **Understood Integration Patterns**: How to combine PlotSenseAI with other tools
✅ **Explored Real-world Applications**: Practical use cases and implementations

## 🚀 Next Steps for Hackathon

### For Participants
1. **Choose Your Path**: Pick the demo that aligns with your interests
2. **Customize and Extend**: Add your own features and improvements
3. **Combine Projects**: Create hybrid applications using multiple demos
4. **Document Your Journey**: Create your own walkthrough for your modifications

### For Organizers
1. **Workshop Planning**: Use these walkthroughs as guided workshop content
2. **Assessment**: Check participant understanding at key checkpoints
3. **Troubleshooting**: Reference common issues and solutions provided
4. **Extension Activities**: Use experiment suggestions for advanced participants

## 🎁 Bonus Challenges

### Beginner Level
- Modify color schemes in visualizations
- Add new cities to the climate dataset
- Change anomaly detection thresholds

### Intermediate Level
- Integrate all three demos into one application
- Add real-time data streaming
- Implement user authentication and data persistence

### Advanced Level
- Create new PlotSenseAI plugin types
- Build mobile-responsive designs
- Add machine learning model comparison features

## 📚 Additional Resources

- **PlotSenseAI Documentation**: [docs.plotsense.ai](https://docs.plotsense.ai)
- **Streamlit Gallery**: [streamlit.io/gallery](https://streamlit.io/gallery)
- **Jupyter Best Practices**: [jupyter-notebook.readthedocs.io](https://jupyter-notebook.readthedocs.io)
- **Data Visualization Principles**: [Visual design principles for data viz](https://github.com/ft-interactive/chart-doctor/tree/master/visual-vocabulary)

---

**Happy Exploring! 🎉**

*Remember: The best way to learn is by doing. Don't hesitate to break things, experiment, and most importantly, have fun with your data!*