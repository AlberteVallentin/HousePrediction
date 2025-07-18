# House Prediction

A machine learning project for house price prediction using various algorithms and models.

## Requirements

- Python 3.11 or newer
- Git

## Installation Guide

### Method 1: Installation with uv

#### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### 2. Clone the project

```bash
git clone <repository-url>
cd HousePrediction
```

#### 3. Install dependencies

```bash
# Install all dependencies from pyproject.toml
uv sync

# Or install from requirements.txt
uv pip install -r requirements.txt
```

#### 4. Activate environment

```bash
# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows
```

### Method 2: Environment Setup with Anaconda

#### 1. Create a new Conda environment

```bash
# Create a new environment with Python 3.11
conda create -n houseprediction python=3.11

# Activate the environment
conda activate houseprediction
```

#### 2. Clone the project

```bash
git clone <repository-url>
cd HousePrediction
```

#### 3. Install dependencies using requirements.txt

Once your Conda environment is activated, install all required packages:

```bash
# Install packages using pip (within your conda environment)
pip install -r requirements.txt
```

#### 4. Register the conda environment with Jupyter

```bash
# Register the conda environment with Jupyter
python -m ipykernel install --user --name houseprediction --display-name "Python (houseprediction)"
```

## Usage

### Start Jupyter Notebook

```bash
# With uv
uv run jupyter notebook

# With conda
conda activate houseprediction
jupyter notebook
```

### Run Streamlit app

```bash
# With uv
uv run streamlit run app.py

# With conda
conda activate houseprediction
streamlit run app.py
```

## Project Structure

```
HousePrediction/
├── notebooks/          # Jupyter notebooks
├── models/             # Trained models
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Exported dependencies
├── uv.lock            # Locked versions for uv
└── README.md          # This file
```

## Dependencies

The project uses the following main libraries:

- **Machine Learning**: catboost, xgboost, lightgbm, scikit-learn, optuna
- **Data Processing**: pandas, numpy, statsmodels
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: jupyter, ipykernel, streamlit
- **Other**: ollama, requests

## Troubleshooting

### uv issues

```bash
# Reinstall dependencies
uv sync --reinstall

# Clear cache
uv cache clean
```

### Conda issues

```bash
# Update conda
conda update conda

# Recreate environment
conda env remove -n houseprediction
conda create -n houseprediction python=3.11
```
