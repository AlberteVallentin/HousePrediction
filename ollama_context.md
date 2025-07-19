# House Price Prediction Project - AI Assistant Context

This file contains verified information about the Ames Housing Price Prediction project for the AI assistant. All numbers and facts in this document are based on actual project files and results.

## Project Overview

**Project Title:** House Price Prediction using Ames Housing Dataset  
**Objective:** Build robust machine learning models to predict house prices  
**Academic Context:** CPH Business Intelligence Exam Project 2025  
**Team:** Data Science/BI Implementation

## Dataset Information (VERIFIED)

### Dataset Size and Structure - COMPLETE ANSWER TEMPLATE

**When asked "What is the size and structure of the dataset?" provide this comprehensive answer:**

The Ames Housing dataset has the following size and structure:

**Dataset Size:**

- **Original training samples:** 1,460 houses
- **Original test samples:** 1,459 houses
- **After preprocessing:** 1,458 training + 1,459 test = 2,917 total samples
- **Outliers removed:** 2 houses (524 and 1299) for data quality issues

**Feature Structure:**

- **Original features:** 81 (training), 80 (test - excluding SalePrice target)
- **After feature engineering:** 191 features in final model
- **Feature types breakdown:**
  - Categorical features: 43 (like MSZoning, Neighborhood, etc.)
  - Numerical features: 38 (like LotArea, GrLivArea, etc.)
  - Engineered features: Created additional features like TotalFlrSF, HouseAge, TotalBaths

**Target Variable:**

- **Variable:** SalePrice (continuous, in USD)
- **Range:** $34,900 - $755,000
- **Mean:** $180,921
- **Distribution:** Right-skewed (requires log transformation)

**Data Quality:**

- **Missing values:** 19 features with missing data (mostly luxury features like pools)
- **Data issues identified:** 34 issues across 31 houses
- **Preprocessing approach:** Conservative outlier removal, engineered features, missing value imputation

### Original Dataset

- **Training samples:** 1,460 houses
- **Test samples:** 1,459 houses
- **Original features:** 81 (training), 80 (test, excluding target)
- **Target variable:** SalePrice (continuous, in USD)
- **Data source:** Ames Housing Dataset (real estate transactions)

### After Preprocessing

- **Training samples:** 1,458 houses (2 outliers removed)
- **Test samples:** 1,459 houses (unchanged)
- **Final engineered features:** 191
- **Outliers removed:** Houses 524 and 1299 (data quality issues, not luxury homes)

### Price Statistics (Original Training Data)

- **Price range:** $34,900 - $755,000
- **Mean price:** $180,921
- **Median price:** $163,000
- **Standard deviation:** $79,442
- **Distribution:** Right-skewed (skewness = 1.88)

## Data Quality and Preprocessing

### Identified Issues

- **Total data quality issues:** 34 issues across 31 houses
- **Main issue types:**
  - Remodel date before construction year
  - Remodel date after house sale
  - Garage year built inconsistencies
  - Missing garage data with garage features present

### Outlier Analysis

- **Statistical outliers (IQR method):** 61 houses (>$340k)
- **Removed for data quality:** Only 2 houses (524, 1299)
- **Rationale:** Preserved natural market variation, only removed clear data errors

### Missing Values

- **Features with missing values:** 19 features
- **Highest missing rates:**
  - PoolQC: 99.7% (most houses have no pools)
  - MiscFeature: 96.4% (rare special features)
  - Alley: 93.8% (not all houses have alley access)
  - Fence: 80.8% (many houses have no fences)

### Feature Engineering

- **Created features:** Multiple engineered features improving model performance
- **Key engineered features:**
  - TotalFlrSF: Combined floor space
  - HouseAge: Age at time of sale
  - BsmtFinSF: Total finished basement area
  - TotalBaths: Combined bathroom count
  - GarageAge: Age of garage
  - HasGarage: Binary garage indicator

## Model Development and Performance

### Model Performance Summary - COMPLETE ANSWER TEMPLATES

**When asked about model performance, provide these specific results:**

**Baseline Model Results (Cross-Validation RMSE):**

1. **CatBoost:** 0.1166 (best baseline)
2. **Ridge:** 0.1218
3. **Gradient Boosting:** 0.1281
4. **LightGBM:** 0.1309
5. **Random Forest:** 0.1418
6. **XGBoost:** 0.1419
7. **ElasticNet:** 0.3174 (poor without tuning)
8. **Lasso:** 0.3203 (poor without tuning)

**After Hyperparameter Optimization (415 trials total):**

1. **Stacking Ensemble:** 0.1114 RMSE (FINAL MODEL)
2. **CatBoost:** 0.1143 RMSE (2.0% improvement)
3. **XGBoost:** 0.1148 RMSE (19% improvement)
4. **Lasso:** 0.1175 RMSE (63% improvement!)
5. **ElasticNet:** 0.1175 RMSE (63% improvement!)

**Final Model Validation Performance:**

- **Model:** Stacking Ensemble (Ridge meta-learner)
- **Cross-validation RMSE:** 0.1114 (log scale)
- **Validation RMSE:** $18,713 (original scale)
- **MAE:** $13,499
- **R² Score:** 0.9366 (explains 93.66% of variance)
- **MAPE:** 8.18%
- **Overall improvement:** +4.51% over best baseline

### Baseline Models Tested (8 algorithms)

1. **CatBoost:** 0.1166 RMSE (best baseline)
2. **Ridge:** 0.1218 RMSE
3. **Gradient Boosting:** 0.1281 RMSE
4. **LightGBM:** 0.1309 RMSE
5. **Random Forest:** 0.1418 RMSE
6. **XGBoost:** 0.1419 RMSE
7. **ElasticNet:** 0.3174 RMSE (poor initial performance)
8. **Lasso:** 0.3203 RMSE (poor initial performance)

### Hyperparameter Optimization

- **Total optimization trials:** 415 trials across all models
- **Method:** Optuna Bayesian optimization with MedianPruner
- **Objective:** Minimize cross-validated RMSE

### Optimized Model Performance

1. **CatBoost:** 0.1143 RMSE (2.0% improvement)
2. **XGBoost:** 0.1148 RMSE (19% improvement)
3. **Lasso:** 0.1175 RMSE (63% improvement!)
4. **ElasticNet:** 0.1175 RMSE (63% improvement!)
5. **Ridge:** 0.1218 RMSE (minimal change)

### Ensemble Methods

1. **Simple Ensemble:** Average of top 4 models
2. **Weighted Ensemble:** Performance-weighted averaging
3. **Stacking Ensemble:** Ridge meta-learner (BEST)

### Final Model Performance

- **Final model:** Stacking Ensemble
- **Cross-validation RMSE:** 0.1114 (log scale)
- **Validation RMSE:** $18,713 (original scale)
- **MAE:** $13,499
- **R² Score:** 0.9366 (93.66% variance explained)
- **MAPE:** 8.18%
- **Performance improvement:** +4.51% over best baseline

## Feature Importance (Top 15)

### Feature Importance Analysis - COMPLETE ANSWER TEMPLATE

**When asked about feature importance, provide this comprehensive ranking:**

Based on permutation importance from the final Stacking Ensemble model:

**Top 15 Most Important Features:**

1. **GrLivArea (0.0200)** - Ground living area in square feet - MOST IMPORTANT
2. **OverallQual (0.0125)** - Overall material and finish quality (1-10 scale)
3. **OverallCond (0.0035)** - Overall condition rating (1-10 scale)
4. **TotalFlrSF (0.0034)** - _ENGINEERED:_ Total floor space (1st + 2nd floor)
5. **LotArea (0.0030)** - Lot size in square feet
6. **HouseAge (0.0030)** - _ENGINEERED:_ Age of house at time of sale
7. **BsmtFinSF (0.0025)** - _ENGINEERED:_ Total finished basement area
8. **TotalBaths (0.0017)** - _ENGINEERED:_ Total bathroom count (full + half)
9. **1stFlrSF (0.0015)** - First floor square footage
10. **GarageArea (0.0015)** - Garage area in square feet
11. **FullBath (0.0014)** - Number of full bathrooms above grade
12. **YearBuilt (0.0013)** - Original construction year
13. **TotRmsAbvGrd (0.0012)** - Total rooms above grade (excludes bathrooms)
14. **GarageCars (0.0011)** - Garage capacity in cars
15. **YearRemodAdd (0.0010)** - Remodel date (or construction date if no remodel)

**Key Insights:**

- **4 out of top 8 features were engineered features** - validates feature engineering approach
- **Living space (GrLivArea) is by far the strongest predictor** (2x more important than next feature)
- **Quality ratings are more predictive than pure size metrics**
- **Engineered features outperformed many original features**

### Feature Engineering Success

- **Created features:** Multiple engineered features improving model performance

1. **GrLivArea:** 0.0200 (Ground living area - most important)
2. **OverallQual:** 0.0125 (Overall quality rating)
3. **OverallCond:** 0.0035 (Overall condition)
4. **TotalFlrSF:** 0.0034 (Engineered: Total floor space)
5. **LotArea:** 0.0030 (Lot size)
6. **HouseAge:** 0.0030 (Engineered: Age at sale)
7. **BsmtFinSF:** 0.0025 (Engineered: Basement finished area)
8. **TotalBaths:** 0.0017 (Engineered: Total bathrooms)
9. **1stFlrSF:** 0.0015 (First floor area)
10. **GarageArea:** 0.0015 (Garage size)
11. **FullBath:** 0.0014 (Number of full bathrooms)
12. **YearBuilt:** 0.0013 (Construction year)
13. **TotRmsAbvGrd:** 0.0012 (Total rooms above ground)
14. **GarageCars:** 0.0011 (Garage capacity)
15. **YearRemodAdd:** 0.0010 (Remodel year)

**Note:** 4 of top 8 features were engineered features, validating feature engineering approach.

## Key Insights and Learnings

### Model Performance Insights

- **Tree-based models** (CatBoost, XGBoost) performed best naturally
- **Linear models** required significant regularization tuning
- **Ensemble methods** provided consistent 1-5% improvements
- **Feature engineering** was crucial for top performance

### Feature Relationships

- **Quality over quantity:** OverallQual more predictive than size alone
- **Living space** is the strongest predictor
- **Engineered features** outperformed many original features
- **Multicollinearity** exists between garage and area features

### Business Value

- **Model reliability:** 93.66% variance explained
- **Prediction accuracy:** ±$18,713 average error
- **Price range coverage:** Works for $47k - $701k homes
- **Practical application:** Suitable for real estate valuation

### Validation Strategy

- **Cross-validation:** 5-fold CV for all models
- **Holdout validation:** Separate validation set
- **Consistency check:** 1.1% difference between train/test means
- **Residual analysis:** Confirms model assumptions

## File Structure and Locations

### Data Files

- **Raw data:** `data/raw/train.csv`, `data/raw/test.csv`
- **Processed:** `data/processed/X_train_final.csv`, etc.
- **Quality logs:** `data/logs/data_quality_issues.csv`

### Model Files

- **Summary:** `models/model_summary.json`
- **Final model:** `models/final_model.joblib`
- **Individual models:** `models/*_Optimized.joblib`
- **Ensembles:** `models/Stacking_Ensemble.joblib`

### Documentation

- **Feature descriptions:** `docs/data_description.txt`
- **Methodology:** `docs/notebook_summary.md`
- **Project overview:** `project_summary.md`

### Notebooks

- **01_exploration.ipynb:** Data analysis and EDA
- **02_preprocessing.ipynb:** Data cleaning and preparation
- **03_feature_engineering.ipynb:** Feature creation and selection
- **04_modeling.ipynb:** Model training and evaluation

## Development Process

### Sprint Structure

1. **Problem Formulation:** Business case and research questions
2. **Data Preparation:** ETL, EDA, and feature engineering
3. **Model Development:** Training, optimization, and ensemble
4. **Deployment:** Streamlit app and interactive tools

### Technology Stack

- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, CatBoost, LightGBM
- **Optimization:** Optuna
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Deployment:** Streamlit
- **AI Assistant:** Ollama with Llama 3.2

## Questions the AI Can Answer

### About Data

- Dataset size, structure, and characteristics
- Price statistics and distributions
- Data quality issues and how they were handled
- Missing value patterns and treatment
- Outlier analysis and removal decisions

### About Models

- Model comparison and performance ranking
- Hyperparameter optimization results
- Ensemble strategy and effectiveness
- Cross-validation methodology
- Final model selection rationale

### About Features

- Feature importance rankings and interpretations
- Engineered feature explanations
- Correlation patterns and multicollinearity
- Feature selection decisions

### About Methodology

- Preprocessing pipeline steps
- Validation strategy and results
- Model assumptions and diagnostics
- Business value and practical applications

## Response Guidelines for AI

### Specific Response Templates

**For "What is the size and structure of the dataset?":**
Use the complete answer template in the Dataset Information section above.

**For model performance questions:**
Always cite specific RMSE numbers and improvements from the Model Performance Summary.

**For feature importance questions:**
Provide the ranked list with importance scores and highlight engineered features.

**For preprocessing questions:**
Mention specific numbers: 2 outliers removed, 19 features with missing values, 34 data quality issues.

### General Guidelines

1. **Use verified data only:** All numbers in this document are confirmed from actual project results
2. **Be specific and structured:** Provide organized, comprehensive answers
3. **Cite exact figures:** Use precise numbers rather than approximations
4. **Highlight key insights:** Don't just list facts, explain their significance
5. **Reference engineered features:** Point out when features were created vs original
6. **Specify sources:** Reference specific files or sections when possible
7. **Acknowledge limitations:** State when information is not available in this context
8. **Focus on insights:** Explain not just what but why and how
9. **Maintain context:** Connect individual facts to overall project goals and learnings
10. **Use formatting:** Structure answers with headers, bullet points, and emphasis for clarity

### Response Quality Standards

- **Comprehensive:** Cover all aspects of the question
- **Precise:** Use exact numbers from verified results
- **Structured:** Organize information logically
- **Insightful:** Explain significance and implications
- **Professional:** Maintain technical accuracy while being accessible

---

\_This context file contains verified information from actual project results. The AI should use this as the authoritative source for all responses about the project.
