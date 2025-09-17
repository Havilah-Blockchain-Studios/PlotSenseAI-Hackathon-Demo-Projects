# Setup Guide 🛠️

This comprehensive setup guide will help you get all PlotSenseAI demo projects running on your system.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space

### Required Tools
- **Python Package Manager**: pip (comes with Python)
- **Version Control**: Git
- **Code Editor**: VS Code, PyCharm, or Jupyter Lab (recommended)

## 🔧 Environment Setup

### 1. Python Installation

#### Windows
```bash
# Download from python.org or use winget
winget install Python.Python.3.11
```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv
```

### 2. Git Installation

#### Windows
```bash
winget install Git.Git
```

#### macOS
```bash
# Git comes with Xcode Command Line Tools
xcode-select --install

# Or using Homebrew
brew install git
```

#### Linux
```bash
sudo apt install git
```

### 3. Verify Installation
```bash
python --version  # Should show Python 3.8+
pip --version     # Should show pip version
git --version     # Should show git version
```

## 📥 Project Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Havilah-Blockchain-Studios/PlotSenseAI-Hackathon-Demo-Projects.git
cd PlotSenseAI-Hackathon-Demo-Projects
```

### 2. Create Virtual Environment
We **strongly recommend** using a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv plotsense-env

# Activate virtual environment
# On Windows:
plotsense-env\Scripts\activate

# On macOS/Linux:
source plotsense-env/bin/activate
```

You should see `(plotsense-env)` in your terminal prompt when activated.

### 3. Install Core Dependencies
```bash
# Install PlotSenseAI
pip install plotsense

# Install common dependencies
pip install pandas numpy matplotlib jupyter
```

## 🎯 Project-Specific Setup

### Project One: ML Explainability Demo

```bash
cd project_one

# Install additional dependencies
pip install ucimlrepo scikit-learn

# Start Jupyter Notebook
jupyter notebook ml_explainability_demo.ipynb
```

**Verification**: The notebook should open in your browser at `http://localhost:8888`

### Project Two: Anomaly Detection Plugin

```bash
cd project_two

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python -m pytest tests/test_detection.py

# Run demo
python examples/demo_anomaly_detection.py
```

**Expected Output**: Console output showing detected anomalies and a visualization.

### Project Three: Data Storytelling App

```bash
cd project_three

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

**Expected Output**: Browser opens to `http://localhost:8501` with the interactive app.

## 🔑 API Keys Setup

### Groq API Key (for Project Three)

1. **Create Account**: Visit [Groq Console](https://console.groq.com)
2. **Generate Key**: Navigate to API Keys section
3. **Copy Key**: Format: `gsk_xxxxxxxxxxxxx`
4. **Use in App**: Enter in the sidebar of Project Three

### PlotSenseAI API (if required)

```bash
# Set environment variable
export PLOTSENSE_API_KEY="your_api_key_here"

# Or add to your shell profile (.bashrc, .zshrc)
echo 'export PLOTSENSE_API_KEY="your_key"' >> ~/.bashrc
```

## 🧪 Testing Your Installation

### Quick Verification Script
Create a file `test_installation.py`:

```python
# Test basic imports
try:
    import plotsense
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ All core packages imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")

# Test PlotSenseAI
try:
    from plotsense import recommender
    print("✅ PlotSenseAI imported successfully!")
except ImportError as e:
    print(f"❌ PlotSenseAI import error: {e}")

print("🎉 Installation verification complete!")
```

Run the test:
```bash
python test_installation.py
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Permission Errors (Windows)
```bash
# Run as administrator or use --user flag
pip install --user plotsense
```

#### 2. Python Command Not Found
```bash
# Try using python3 instead of python
python3 --version
python3 -m pip install plotsense
```

#### 3. Virtual Environment Issues
```bash
# Deactivate and recreate
deactivate
rm -rf plotsense-env
python -m venv plotsense-env
source plotsense-env/bin/activate  # or plotsense-env\Scripts\activate on Windows
```

#### 4. Jupyter Notebook Not Starting
```bash
# Install jupyter explicitly
pip install jupyter

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

#### 5. Port Already in Use (Streamlit)
```bash
# Use different port
streamlit run app.py --server.port 8502
```

#### 6. Package Version Conflicts
```bash
# Create fresh environment
pip freeze > old_requirements.txt
deactivate
rm -rf plotsense-env
python -m venv plotsense-env
source plotsense-env/bin/activate
pip install plotsense
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for detailed error messages
2. **Update packages**: `pip install --upgrade plotsense`
3. **Search issues**: Check our [GitHub Issues](https://github.com/Havilah-Blockchain-Studios/PlotSenseAI-Hackathon-Demo-Projects/issues)
4. **Ask for help**: Create a new issue with:
   - Your operating system
   - Python version
   - Error message
   - Steps to reproduce

## 🎯 Next Steps

Once your environment is set up:

1. **Read the tutorials**: Check out [TUTORIALS.md](./TUTORIALS.md)
2. **Explore projects**: Start with Project One for PlotSenseAI basics
3. **Join the community**: Participate in discussions and contribute
4. **Build something awesome**: Use these demos as inspiration for your own projects!

## 📚 Additional Resources

- [PlotSenseAI Documentation](https://docs.plotsense.ai)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)

---

**Happy coding! 🚀**