# PlotSense Anomaly Detection Plugin

## 📌 Overview
This is a **demo plugin** for PlotSense AI that shows how developers can extend it with anomaly detection functionality.  
We implement a simple **Z-score based anomaly detector**, integrate it with PlotSense, and provide visualization of results.

---

## ⚙️ Cloning and Installation
Clone and install requirements:
```bash
git clone https://github.com/Havilah-Blockchain-Studios/PlotSenseAI-Hackathon-Demo-Projects.git
cd project_two
pip install -r requirements.txt
```

## 🚀 Usage Example
Run the demo script:
```bash
python -m examples/demo_anomaly_detection.py
```
Expected Output:
Console printout of detected anomalies.
A PlotSense-generated visualization of the dataset.

## 📂 Project Structure
```markdown

plotsense-anomaly-plugin/
├── plotsense_anomaly/
│   ├── __init__.py      # tells Python plotsense_anomaly is a package (it can even be empty)
│   ├── detection.py            # anomaly detection logic
│   └── visualization.py        # connect anomalies → PlotSense
├── examples/
│   └── demo_anomaly_detection.py   # working example
├── tests/
│   └── test_detection.py          # lightweight tests
├── README.md
└── requirements.txt

```

## 🧪 Running Tests
Run lightweight tests with:
```bash
python -m pytest tests/test_detection.py
```

