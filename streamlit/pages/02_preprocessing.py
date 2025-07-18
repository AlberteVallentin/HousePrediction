import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to access utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import (
    load_raw_data, 
    create_combined_dataset, 
    load_processed_data,
    load_feature_descriptions_cached,
    get_numerical_features,
    get_categorical_features
)
from utils.visualizations import (
    create_missing_values_bar_chart,
    create_target_distribution_plot,
    create_scatter_plot,
    create_before_after_comparison,
    create_enhanced_outlier_plot
)

# Page configuration
st.set_page_config(
    page_title="Data Preprocessing",
    layout="wide"
)

st.title("Data Preprocessing")
st.markdown("---")

st.markdown("""
This page demonstrates the systematic preprocessing pipeline that transforms raw data into model-ready format.
The process includes feature type correction, outlier removal, missing value treatment, and data quality corrections.
""")

# Load data
@st.cache_data
def load_preprocessing_data():
    """Load raw and processed data for preprocessing analysis."""
    # Load raw data
    df_combined, df_train, df_test = create_combined_dataset()
    
    # Load processed data
    processed_data = load_processed_data()
    
    return df_combined, df_train, df_test, processed_data

df_combined, df_train, df_test, processed_data = load_preprocessing_data()

if df_combined is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Section 1: Raw Data Overview
st.header("1. Raw Data Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Samples", len(df_train))
with col2:
    st.metric("Test Samples", len(df_test))
with col3:
    st.metric("Total Missing Values", df_combined.isnull().sum().sum())

# Data types before processing
st.subheader("Data Types Before Processing")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Data Type Distribution**")
    dtype_counts = df_combined.dtypes.value_counts()
    fig = px.bar(x=[str(dtype) for dtype in dtype_counts.index], y=dtype_counts.values, 
                 title="Data Types in Raw Dataset")
    fig.update_layout(xaxis_title="Data Type", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Missing Values Pattern**")
    missing_counts = df_combined.isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    st.metric("Features with Missing Values", len(missing_features))
    st.metric("Total Missing Values", missing_counts.sum())

# Section 2: Feature Type Corrections
st.header("2. Feature Type Corrections")

st.markdown("""
Three features were identified as misclassified ordinal variables that need conversion from integer to categorical:
- **OverallQual**: Overall quality rating (1-10 scale)
- **OverallCond**: Overall condition rating (1-10 scale)  
- **MSSubClass**: Dwelling type categories
""")

# Show feature correction example
st.subheader("Feature Type Correction Example")

# Create example data for demonstration
correction_data = {
    'Feature': ['OverallQual', 'OverallCond', 'MSSubClass'],
    'Original Type': ['int64', 'int64', 'int64'],
    'Corrected Type': ['Ordered Categorical', 'Ordered Categorical', 'Categorical'],
    'Unique Values': ['1-10', '1-9', '20,30,40,45,50,60,70,75,80,85,90,120,150,160,180,190']
}

correction_df = pd.DataFrame(correction_data)
st.dataframe(correction_df, use_container_width=True)

# Section 3: Outlier Detection and Removal
st.header("3. Outlier Detection and Removal")

st.markdown("""
Two problematic outliers were identified in the analysis:
- **ID 524**: 4,676 sqft house sold for $184,750 (OverallQual=10)
- **ID 1299**: 5,642 sqft house sold for $160,000 (OverallQual=10)

These represent data quality issues - luxury homes with maximum quality ratings 
sold at prices inconsistent with their characteristics. They appear to be 
partial sales of incomplete luxury properties.
""")

# Enhanced Outlier Analysis
st.subheader("Enhanced Outlier Analysis")

if 'SalePrice' in df_train.columns and 'GrLivArea' in df_train.columns:
    st.markdown("""
    Comprehensive outlier analysis showing the impact of problematic data points.
    This enhanced visualization includes scatter plots, box plots, and impact analysis.
    """)
    
    # Create enhanced outlier plot with specific IDs
    outlier_plot = create_enhanced_outlier_plot(df_train, 'GrLivArea', 'SalePrice', outlier_ids=[524, 1299])
    st.plotly_chart(outlier_plot, use_container_width=True)
    
    # Show outlier details
    st.subheader("Outlier Analysis Legend")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("🔵 **Normal Points**")
        st.write("Properties within normal range")
        
    with col2:
        st.markdown("🔴 **IQR Outliers**")
        st.write("Statistical outliers detected by IQR method")
        
    with col3:
        st.markdown("🟡 **Removed Outliers**")
        st.write("Data quality issues removed from dataset")
    
    st.subheader("Removed Outlier Details")
    outlier_details = {
        'Property ID': [524, 1299],
        'Living Area (sqft)': [4676, 5642],
        'Sale Price ($)': [184750, 160000],
        'Overall Quality': [10, 10],
        'Issue': ['Partial sale of incomplete luxury property', 'Partial sale of incomplete luxury property']
    }
    
    outlier_df = pd.DataFrame(outlier_details)
    st.dataframe(outlier_df, use_container_width=True)
    
    st.info("💡 **Key Insight**: The gold diamond markers show the specific outliers we removed due to data quality issues. These are different from the red statistical outliers detected by the IQR method.")
else:
    st.info("Enhanced outlier analysis not available - required columns not found.")

# Impact of outlier removal
st.subheader("Impact of Outlier Removal")

col1, col2 = st.columns(2)
with col1:
    st.metric("Original Training Samples", "1,460")
    st.metric("After Outlier Removal", "1,458")
    st.metric("Samples Removed", "2 (0.14%)")

with col2:
    st.markdown("**Impact Assessment:**")
    st.write("• Minimal impact on overall distribution")
    st.write("• Removed problematic data quality issues")
    st.write("• Preserved 99.86% of training data")
    st.write("• Improved model reliability")

# Section 4: Missing Value Treatment
st.header("4. Missing Value Treatment")

st.markdown("""
Systematic three-tier approach to missing value treatment:
1. **Architectural Absence**: Features like pools, garages, basements filled with 'None'
2. **Domain-Specific Imputation**: LotFrontage filled with neighborhood medians
3. **Mode/Median Imputation**: Remaining features filled with appropriate central tendency
""")

# Missing values overview
st.subheader("Missing Values Analysis")

missing_summary = {
    'Category': ['High Missingness (>80%)', 'Medium Missingness (5-80%)', 'Low Missingness (<5%)'],
    'Features': ['PoolQC, MiscFeature, Alley, Fence', 'MasVnrType, FireplaceQu, LotFrontage', 'Various basement and garage features'],
    'Treatment': ['Fill with "None"', 'Domain-specific imputation', 'Mode/median imputation']
}

missing_df = pd.DataFrame(missing_summary)
st.dataframe(missing_df, use_container_width=True)

# Top missing features visualization
st.subheader("Top Features with Missing Values")

# Actual missing values visualization
if df_combined is not None:
    missing_bar_chart = create_missing_values_bar_chart(df_combined)
    if missing_bar_chart:
        st.plotly_chart(missing_bar_chart, use_container_width=True)
    else:
        st.info("No missing values visualization available.")
else:
    st.info("Missing values analysis requires the actual dataset to be loaded.")

# Before/After Missing Values Comparison
st.subheader("Before/After Missing Values Treatment")
st.markdown("""
Compare the missing values situation before and after preprocessing.
The 'after' dataset shows zero missing values across all features.
""")

# Create before/after comparison
if processed_data and 'train_cleaned' in processed_data and 'test_cleaned' in processed_data:
    # Use the cleaned datasets (preprocessing output)
    train_cleaned_temp = processed_data['train_cleaned'].drop('SalePrice', axis=1, errors='ignore')
    test_cleaned_temp = processed_data['test_cleaned']
    
    # Ensure both datasets have the same columns
    common_columns = train_cleaned_temp.columns.intersection(test_cleaned_temp.columns)
    train_cleaned_temp = train_cleaned_temp[common_columns]
    test_cleaned_temp = test_cleaned_temp[common_columns]
    
    # Combine cleaned datasets
    combined_cleaned_temp = pd.concat([train_cleaned_temp, test_cleaned_temp], ignore_index=True)
    
    
    before_after_plot = create_before_after_comparison(df_combined, combined_cleaned_temp)
    st.plotly_chart(before_after_plot, use_container_width=True)

# Section 5: Data Quality Corrections
st.header("5. Data Quality Corrections")

st.markdown("""
Systematic correction of data quality issues identified through validation:
""")

# Data quality issues summary
quality_issues = {
    'Issue': [
        'Remodel date before construction',
        'Pool area without quality rating',
        'No kitchen above grade',
        'Basement area without quality',
        'Garage year typo (2207 → 2007)',
        'Missing miscellaneous feature value'
    ],
    'Houses_Affected': [1, 3, 3, 2, 1, 1],
    'Resolution': [
        'Set YearRemodAdd = YearBuilt',
        'Set PoolArea = 0',
        'Set KitchenAbvGr = 1',
        'Infer quality from age/condition',
        'Correct typo to 2007',
        'Set MiscFeature = "Othr"'
    ]
}

quality_df = pd.DataFrame(quality_issues)
st.dataframe(quality_df, use_container_width=True)

# Section 6: Data Type Optimization
st.header("6. Data Type Optimization")

st.markdown("""
Optimized data types for improved model performance:
- Converted float features with whole numbers to integers
- Preserved decimal precision where needed (LotFrontage)
- Maintained categorical encodings
""")

# Show data type optimization results
st.subheader("Data Type Optimization Results")

optimization_data = {
    'Category': ['Integer Features', 'Float Features', 'Categorical Features', 'Total Features'],
    'Before': [23, 11, 46, 80],
    'After': [33, 1, 46, 80],
    'Change': ['+10', '-10', '0', '0']
}

opt_df = pd.DataFrame(optimization_data)
st.dataframe(opt_df, use_container_width=True)

# Section 7: Before/After Comparison
st.header("7. Before/After Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data")
    st.metric("Training Samples", "1,460")
    st.metric("Missing Values", "7,829")
    st.metric("Data Quality Issues", "34")
    st.metric("Float Features", "11")

with col2:
    st.subheader("Processed Data")
    st.metric("Training Samples", "1,458")
    st.metric("Missing Values", "0")
    st.metric("Data Quality Issues", "0")
    st.metric("Float Features", "1")

# Section 8: Validation Results
st.header("8. Validation Results")

st.markdown("""
**Final Preprocessing Pipeline Validation:**
- **Missing Values**: 0 remaining across all features
- **Data Types**: Consistent between train/test sets
- **Feature Count**: 80 features preserved
- **Outlier Removal**: 2 problematic cases removed (0.14% of data)
- **Data Quality**: All 34 identified issues resolved
- **Type Optimization**: 10 features converted to integer
""")

# Show processed data sample if available
if processed_data and 'train_cleaned' in processed_data:
    st.subheader("Processed Data Sample")
    st.dataframe(processed_data['train_cleaned'].head(10), use_container_width=True)

# Section 9: Interactive Data Quality Dashboard
st.header("9. Interactive Data Quality Dashboard")

# Create combined cleaned dataset for fair comparison
combined_cleaned = None
if processed_data and 'train_cleaned' in processed_data and 'test_cleaned' in processed_data:
    # Use the cleaned datasets (preprocessing output)
    train_cleaned = processed_data['train_cleaned'].drop('SalePrice', axis=1, errors='ignore')
    test_cleaned = processed_data['test_cleaned']
    
    # Ensure both datasets have the same columns
    common_columns = train_cleaned.columns.intersection(test_cleaned.columns)
    train_cleaned = train_cleaned[common_columns]
    test_cleaned = test_cleaned[common_columns]
    
    # Combine cleaned datasets
    combined_cleaned = pd.concat([train_cleaned, test_cleaned], ignore_index=True)

if processed_data and 'train_cleaned' in processed_data:
    st.markdown("""
    Interactive dashboard showing the comprehensive preprocessing pipeline results.
    Toggle between different aspects of the data quality improvements.
    """)
    
    # Create tabs for different quality metrics
    tab1, tab2 = st.tabs(["Missing Values", "Data Types"])
    
    with tab1:
        st.subheader("Missing Values Treatment Progress")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            before_missing = df_combined.isnull().sum().sum()
            st.metric("Before Treatment", f"{before_missing:,} missing values")
        with col2:
            after_missing = processed_data['train_cleaned'].isnull().sum().sum()
            st.metric("After Treatment", f"{after_missing:,} missing values")
        with col3:
            improvement = ((before_missing - after_missing) / before_missing * 100)
            st.metric("Improvement", f"{improvement:.1f}%")
        
        # Feature-wise missing values comparison
        if before_missing > 0:
            missing_comparison = pd.DataFrame({
                'Feature': df_combined.columns,
                'Before': df_combined.isnull().sum(),
                'After': processed_data['train_cleaned'].isnull().sum().reindex(df_combined.columns, fill_value=0)
            })
            missing_comparison = missing_comparison[missing_comparison['Before'] > 0]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Before',
                x=missing_comparison['Feature'],
                y=missing_comparison['Before'],
                marker_color='red',
                opacity=0.7
            ))
            fig.add_trace(go.Bar(
                name='After',
                x=missing_comparison['Feature'],
                y=missing_comparison['After'],
                marker_color='green',
                opacity=0.7
            ))
            
            fig.update_layout(
                title='Missing Values: Before vs After Treatment',
                xaxis_title='Features',
                yaxis_title='Missing Count',
                barmode='group',
                xaxis_tickangle=-45,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Data Type Optimization")
        
        # Data type comparison
        before_types = df_combined.dtypes.value_counts()
        after_types = processed_data['train_cleaned'].dtypes.value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Optimization:**")
            for dtype, count in before_types.items():
                st.write(f"• {dtype}: {count} features")
        
        with col2:
            st.markdown("**After Optimization:**")
            for dtype, count in after_types.items():
                st.write(f"• {dtype}: {count} features")
        
        # Data type changes visualization
        type_changes = pd.DataFrame({
            'Data Type': ['int64', 'float64', 'object'],
            'Before': [before_types.get(dtype, 0) for dtype in ['int64', 'float64', 'object']],
            'After': [after_types.get(dtype, 0) for dtype in ['int64', 'float64', 'object']]
        })
        
        fig = px.bar(type_changes, 
                     x='Data Type', 
                     y=['Before', 'After'],
                     title='Data Type Distribution Changes',
                     barmode='group',
                     height=400)
        st.plotly_chart(fig, use_container_width=True)
    
else:
    st.info("Processed data comparison requires both original and processed datasets to be loaded.")

# Section 10: Key Insights
st.header("10. Key Insights")

st.markdown("""
**Preprocessing Results:**
- **Data Quality Issues Resolved**: 34 specific issues across 31 houses corrected
- **Outlier Removal**: 2 problematic properties (IDs 524, 1299) removed from training set
- **Missing Value Treatment**: 7,829 missing values addressed using domain-specific strategies
- **Type Corrections**: 3 ordinal features (OverallQual, OverallCond, MSSubClass) properly classified

**Specific Corrections Applied:**
- **Timeline Issues**: Fixed 1 remodel-before-construction case and 1 garage typo (2207→2007)
- **Feature Consistency**: Corrected 3 houses with PoolArea>0 but missing PoolQC
- **Structural Logic**: Fixed 3 houses with KitchenAbvGr=0 but existing KitchenQual
- **Data Quality**: Resolved basement area without quality ratings (2 cases)

**Three-Tier Missing Value Strategy:**
- **Architectural Absence**: PoolQC, MiscFeature, Alley, Fence filled with 'None' (represents absence)
- **Domain-Specific**: LotFrontage filled with neighborhood medians (real estate expertise)
- **Statistical**: Remaining features filled with mode/median based on data type

**Final Dataset State:**
- **Training Samples**: 1,458 (removed 2 outliers from original 1,460)
- **Missing Values**: 0 across all features (100% complete)
- **Data Types**: Optimized from 11 float features to 1 (improved memory efficiency)
- **Consistency**: Train/test preprocessing pipeline maintains identical feature structure

**Model Readiness Achieved:**
- Zero missing values enable all ML algorithms
- Consistent data types prevent training/inference mismatches
- Domain-appropriate missing value treatment preserves real estate business logic
- Quality corrections eliminate erroneous pattern learning opportunities
""")

# Footer
st.markdown("---")
st.markdown("*This preprocessing pipeline mirrors the systematic approach from the 02_preprocessing.ipynb notebook*")