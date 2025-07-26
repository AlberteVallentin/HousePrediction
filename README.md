# Intelligent House Price Prediction System

A comprehensive Business Intelligence solution leveraging machine learning for automated residential property price prediction with professional-grade accuracy.

## Project Overview

This project develops an intelligent machine learning system that predicts residential property prices by analyzing property characteristics from the Ames Housing dataset. Through systematic application of data analytics and AI technologies, we created a robust solution that addresses pricing transparency challenges in real estate markets.

# Problem Formulation

## Project Title

**Intelligent House Price Prediction System - A Machine Learning Approach to Real Estate Valuation**

## Problem Statement

In real estate markets, pricing a home accurately is one of the most critical — and often most subjective — parts of the transaction process. Buyers, sellers, and agents alike frequently rely on intuition or generic market trends rather than the unique characteristics of each property and its surroundings. This project addresses that challenge by building a Business Intelligence solution that uncovers patterns in historical housing data to explain how factors like location, condition, size, and amenities influence sale prices. By analyzing structured housing data from Ames, Iowa, we aim to increase transparency in the pricing process and support data-driven decision-making for all market participants.

## Relevance

This challenge is important because housing is one of the largest financial decisions most people make, yet the pricing process often lacks transparency. By integrating data analytics and machine learning, this project proposes a practical and evidence-based approach to improve fairness and accuracy in property valuation.

## Research Questions

**Primary Research Question:**

- How can machine learning techniques be applied to predict residential property prices, and what level of accuracy can be achieved using property characteristics as predictive features?

**Secondary Research Questions:**

- Which property characteristics have the greatest influence on house price prediction, and how can we quantify their relative importance in the valuation process?
- How do different categories of machine learning algorithms compare in terms of prediction accuracy and model reliability for real estate valuation?
- To what extent can automated machine learning models provide practical value for real estate stakeholders compared to traditional property valuation methods?

## Hypotheses

**H1**
Physical property characteristics related to size and quality will emerge as the strongest predictors of house prices, reflecting fundamental real estate valuation principles where larger, higher-quality properties command premium prices.

**H2**
Advanced machine learning algorithms will outperform traditional linear models in real estate price prediction due to the complex, non-linear relationships between multiple property characteristics and market values.

**H3**
Combining multiple machine learning approaches will achieve superior prediction accuracy compared to relying on any single modeling technique, as different algorithms may capture different aspects of the price-prediction relationship.

## Key Results

- **RMSE:** $18,713 (8.18% MAPE)
- **R² Score:** 0.9366 (93.66% variance explained)
- **Models Tested:** 11 algorithms with 415 optimization trials
- **Final Model:** Stacking ensemble with meta-learning
- **Feature Engineering:** 81 → 191 features

## Dataset Information

**Link to Dataset** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- **Training Samples:** 1,458 properties (after preprocessing)
- **Test Samples:** 1,459 properties
- **Price Range:** $47,690 - $701,555
- **Features:** 191 engineered characteristics
- **Source:** Ames Housing Dataset (2006-2010)

## Project Structure

```
HousePrediction/
├── README.md
├── requirements.txt
├── docs/
│   ├── problem-statement.md
│   ├── notebook_summary.md
│   ├── conclusion.md
│   └── data_description.txt
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── models/
│   ├── final_model.joblib
│   ├── model_summary.json
│   └── feature_order.json
├── streamlit/
│   ├── app.py
│   ├── pages/
│   │   ├── 01_exploration.py
│   │   ├── 02_preprocessing.py
│   │   ├── 03_feature_engineering.py
│   │   ├── 04_modeling.py
│   │   ├── 05_prediction.py
│   │   └── 06_ai_chat.py
│   └── utils/
│       ├── ai_chat_utils.py
│       ├── data_loader.py
│       ├── model_utils.py
│       └── visualizations.py
└── submissions/
```

- Business value demonstration

### Model Performance Hierarchy

1. **Stacking Ensemble:** 0.1114 RMSE (Best)
2. **CatBoost:** 0.1143 RMSE
3. **XGBoost:** 0.1148 RMSE
4. **LightGBM:** 0.1172 RMSE
5. **Lasso:** 0.1175 RMSE

## Validated Hypotheses

**H1:** Size and quality characteristics emerged as strongest predictors  
**H2:** Advanced ML algorithms outperformed linear models by 3-7%  
**H3:** Ensemble methods achieved 4.51% improvement over baselines

## Technologies Used

- **Python 3.11+**
- **Machine Learning:** scikit-learn, CatBoost, XGBoost, LightGBM
- **Data Analysis:** pandas, numpy, matplotlib, seaborn
- **Web Application:** Streamlit
- **Model Persistence:** joblib
- **Development:** Jupyter Notebooks

## Business Impact

- **Instant Predictions:** Seconds vs. days for traditional appraisals
- **Objective Valuation:** Eliminates human bias and subjectivity
- **Transparent Logic:** Feature importance explains pricing decisions
- **Scalable Solution:** Handles diverse property types and price ranges
- **Stakeholder Value:** Benefits buyers, sellers, agents, and investors

## Model Validation

- **Generalization:** 1.1% difference between train/test predictions
- **Kaggle Competition:** 0.11929 public score (external validation)
- **Residual Analysis:** Near-zero mean, homoscedastic variance
- **Cross-Validation:** Consistent performance across folds
- **Price Coverage:** Reliable across $47K - $701K range

## Documentation

Detailed documentation available in the `docs/` folder:

- [Problem Statement](docs/problem-statement.md)
- [Notebook Summary](docs/notebook_summary.md)
- [Data Description](docs/data_description.txt)
- [Conclusion](docs/conclusion.md)

## Summary of the Project

This project successfully demonstrates how machine learning can transform real estate valuation by providing accurate, data-driven property price predictions. The developed system not only enhances transparency and fairness in pricing but also empowers stakeholders with actionable insights for informed decision-making.
Read the fulll conclusion

## Team

Alberte Mary Wahlstrøm Vallentin

## Installation Guide

### Method 1: Installation with uv

#### 1. Clone the project

```bash
git clone https://github.com/AlberteVallentin/HousePrediction.git
cd HousePrediction
```

#### 2. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
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

#### 1. Clone the project

```bash
git clone https://github.com/AlberteVallentin/HousePrediction.git
cd HousePrediction
```

#### 2. Create a new Conda environment

```bash
# Create a new environment with Python 3.11
conda create -n houseprediction python=3.11

# Activate the environment
conda activate houseprediction
```

#### 3. Install dependencies using requirements.txt

Once your Conda environment is activated, install all required packages from our clean, organized requirements file:

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
cd streamlit
uv run streamlit run app.py

# With conda
conda activate houseprediction
cd streamlit
streamlit run app.py
```

### AI Chat Assistant (Optional)

The project includes an AI-powered chat assistant using Ollama for project Q&A:

1. **Install Ollama:** Download from ollama.ai
2. **Pull a model:** ollama pull llama3.2
3. **Access chat:** Navigate to the "AI Assistant" page in the Streamlit app

The AI assistant can answer questions about the project methodology, results, and implementation details.

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
