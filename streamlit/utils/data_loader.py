import pandas as pd
import numpy as np
import streamlit as st
import os
import sys

# Add the parent directory to access notebooks utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from notebooks.data_description_parser import load_feature_descriptions

@st.cache_data
def load_raw_data():
    """Load raw training and test data."""
    try:
        train_path = os.path.join(os.path.dirname(__file__), '../../data/raw/train.csv')
        test_path = os.path.join(os.path.dirname(__file__), '../../data/raw/test.csv')
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        return df_train, df_test
    except Exception as e:
        st.error(f"Error loading raw data: {str(e)}")
        return None, None

@st.cache_data
def load_processed_data():
    """Load processed data if available."""
    try:
        processed_path = os.path.join(os.path.dirname(__file__), '../../data/processed/')
        
        files = {
            'X_train': 'X_train_final.csv',
            'X_test': 'X_test_final.csv', 
            'y_train': 'y_train_final.csv',
            'train_cleaned': 'train_cleaned.csv',
            'test_cleaned': 'test_cleaned.csv'
        }
        
        data = {}
        for key, filename in files.items():
            file_path = os.path.join(processed_path, filename)
            if os.path.exists(file_path):
                data[key] = pd.read_csv(file_path)
        
        return data
    except Exception as e:
        st.error(f"Error loading processed data: {str(e)}")
        return {}

@st.cache_data
def create_combined_dataset():
    """Create combined dataset for analysis."""
    df_train, df_test = load_raw_data()
    
    if df_train is not None and df_test is not None:
        # Create combined dataset
        df_combined = pd.concat([
            df_train.drop('SalePrice', axis=1), 
            df_test
        ], ignore_index=True)
        df_combined['dataset_source'] = ['train']*len(df_train) + ['test']*len(df_test)
        
        return df_combined, df_train, df_test
    
    return None, None, None

@st.cache_data
def load_feature_descriptions_cached():
    """Load feature descriptions with caching."""
    try:
        descriptions_path = os.path.join(os.path.dirname(__file__), '../../docs/data_description.txt')
        return load_feature_descriptions(descriptions_path)
    except Exception as e:
        st.error(f"Error loading feature descriptions: {str(e)}")
        return None

@st.cache_data
def load_data_quality_log():
    """Load data quality issues log."""
    try:
        log_path = os.path.join(os.path.dirname(__file__), '../../data/logs/data_quality_issues.csv')
        if os.path.exists(log_path):
            return pd.read_csv(log_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data quality log: {str(e)}")
        return pd.DataFrame()

def get_numerical_features(df):
    """Get numerical features from dataframe."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Remove non-meaningful features
    remove_cols = ['Id']
    return [col for col in numerical_cols if col not in remove_cols]

def get_categorical_features(df):
    """Get categorical features from dataframe."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove non-meaningful features
    remove_cols = ['Id', 'dataset_source']
    return [col for col in categorical_cols if col not in remove_cols]

def get_feature_info(feature_name):
    """Get detailed information about a specific feature."""
    descriptions = load_feature_descriptions_cached()
    if descriptions is not None:
        feature_info = descriptions[descriptions['Feature'] == feature_name]
        if not feature_info.empty:
            return feature_info.iloc[0]
    return None