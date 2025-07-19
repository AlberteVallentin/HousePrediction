# HousePrediction Project Structure

```
HousePrediction/
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── ollama_setup_guide.md
├── project_summary.md
│
├── data/
│   ├── raw/
│   │   ├── test.csv
│   │   └── train.csv
│   ├── processed/
│   │   ├── X_test_final.csv
│   │   ├── X_train_final.csv
│   │   ├── test_cleaned.csv
│   │   ├── train_cleaned.csv
│   │   └── y_train_final.csv
│   └── logs/
│       └── data_quality_issues.csv
│
├── docs/
│   ├── Exam-Project.pdf
│   ├── data_description.txt
│   ├── notebook_summary.md
│   └── problem-statement.md
│
├── models/
│   ├── CatBoost_Optimized.joblib
│   ├── CatBoost_Optimized_metadata.json
│   ├── ElasticNet_Optimized.joblib
│   ├── ElasticNet_Optimized_metadata.json
│   ├── GradientBoosting_Optimized.joblib
│   ├── GradientBoosting_Optimized_metadata.json
│   ├── Lasso_Optimized.joblib
│   ├── Lasso_Optimized_metadata.json
│   ├── LightGBM_Optimized.joblib
│   ├── LightGBM_Optimized_metadata.json
│   ├── RandomForest_Optimized.joblib
│   ├── RandomForest_Optimized_metadata.json
│   ├── Ridge_Optimized.joblib
│   ├── Ridge_Optimized_metadata.json
│   ├── Simple_Ensemble.joblib
│   ├── Simple_Ensemble_metadata.json
│   ├── Stacking_Ensemble.joblib
│   ├── Stacking_Ensemble_metadata.json
│   ├── Weighted_Ensemble.joblib
│   ├── Weighted_Ensemble_metadata.json
│   ├── XGBoost_Optimized.joblib
│   ├── XGBoost_Optimized_metadata.json
│   ├── feature_order.json
│   ├── final_model.joblib
│   ├── final_model_metadata.json
│   └── model_summary.json
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── data_description_parser.py
│   └── catboost_info/
│       ├── catboost_training.json
│       ├── learn_error.tsv
│       ├── time_left.tsv
│       ├── learn/
│       │   └── events.out.tfevents
│       └── tmp/
│
├── streamlit/
│   ├── app.py
│   ├── pages/
│   │   ├── 01_exploration.py
│   │   ├── 02_preprocessing.py
│   │   ├── 03_feature_engineering.py
│   │   ├── 04_modeling.py
│   │   ├── 05_prediction.py
│   │   ├── 06_ai_chat.py
│   │   ├── 07_t.py
│   │   ├── 08_p.py
│   │   ├── 09_f.py
│   │   └── 10_m.py
│   └── utils/
│       ├── ai_chat_utils.py
│       ├── data_loader.py
│       ├── model_utils.py
│       └── visualizations.py
│
└── submissions/
    ├── submission_CatBoost_XGBoost_80_20.csv
    ├── submission_Simple_Ensemble.csv
    ├── submission_Simple_Ensemble1.csv
    ├── submission_Simple_Ensemble2.csv
    ├── submission_Simple_Ensemble22.csv
    ├── submission_Simple_Ensemble4.csv
    ├── submission_Stacking_Ensemble5.csv
    ├── submission_Stacking_Ensemble6.csv
    ├── submission_Stacking_Ensemblenew.csv
    ├── submission_Stacking_Ensemblenew1.csv
    ├── submission_Stacking_Ensemblenew2.csv
    ├── submission_Stacking_Ensemblenew3.csv
    ├── submission_Stacking_Ensemblenew4.csv
    ├── submission_elastic_net_individual.csv
    └── submission_ensemble_optimized.csv
```

## Project Overview

This is a machine learning project for house price prediction with the following key components:

- **data/**: Contains raw datasets, processed data files, and data quality logs
- **docs/**: Project documentation including exam requirements and problem statements
- **models/**: Trained machine learning models with metadata and ensemble configurations
- **notebooks/**: Jupyter notebooks for data exploration, preprocessing, feature engineering, and modeling
- **streamlit/**: Web application with interactive pages and utility functions
- **submissions/**: CSV files containing model predictions for various ensemble approaches
