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

These represent data quality issues where luxury homes with maximum quality ratings 
were sold at prices inconsistent with their characteristics.
""")

# Enhanced Outlier Analysis
st.subheader("Enhanced Outlier Analysis")

if 'SalePrice' in df_train.columns and 'GrLivArea' in df_train.columns:
    st.markdown("""
    Comprehensive outlier analysis showing the impact of problematic data points.
    This enhanced visualization includes scatter plots, box plots, and impact analysis.
    """)
    
    # Create enhanced outlier plot
    outlier_plot = create_enhanced_outlier_plot(df_train, 'GrLivArea', 'SalePrice', [524, 1299])
    st.plotly_chart(outlier_plot, use_container_width=True)
    
    # Show outlier details
    st.subheader("Outlier Details")
    outlier_details = {
        'Property ID': [524, 1299],
        'Living Area (sqft)': [4676, 5642],
        'Sale Price ($)': [184750, 160000],
        'Overall Quality': [10, 10],
        'Issue': ['Luxury home sold below market', 'Very large home sold below market']
    }
    
    outlier_df = pd.DataFrame(outlier_details)
    st.dataframe(outlier_df, use_container_width=True)
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
    st.markdown("**Minimal Impact on Distribution:**")
    st.text("• Mean price: $180,921 → $180,933")
    st.text("• Median price: $163,000 → $163,000")
    st.text("• Skewness: 1.8829 → 1.8813")

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
if processed_data and 'train_cleaned' in processed_data:
    before_after_plot = create_before_after_comparison(df_combined, processed_data['train_cleaned'])
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

if processed_data and 'train_cleaned' in processed_data:
    st.markdown("""
    Interactive dashboard showing the comprehensive preprocessing pipeline results.
    Toggle between different aspects of the data quality improvements.
    """)
    
    # Create tabs for different quality metrics
    tab1, tab2, tab3 = st.tabs(["Missing Values", "Data Types", "Statistical Changes"])
    
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
    
    with tab3:
        st.subheader("Statistical Summary Changes")
        
        # Compare numerical feature statistics
        numerical_cols = df_combined.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            # Select a few key numerical features for comparison
            key_features = ['SalePrice', 'LotArea', 'GrLivArea', 'OverallQual']
            available_features = [col for col in key_features if col in numerical_cols]
            
            if available_features:
                selected_feature = st.selectbox(
                    "Select feature for statistical comparison:",
                    available_features
                )
                
                # Calculate statistics
                before_stats = df_combined[selected_feature].describe()
                after_stats = processed_data['train_cleaned'][selected_feature].describe()
                
                # Create comparison table
                comparison_stats = pd.DataFrame({
                    'Before': before_stats,
                    'After': after_stats,
                    'Change': after_stats - before_stats
                })
                
                st.dataframe(comparison_stats, use_container_width=True)
                
                # Distribution comparison
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Before Processing', 'After Processing')
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=df_combined[selected_feature].dropna(),
                        nbinsx=30,
                        name='Before',
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=processed_data['train_cleaned'][selected_feature].dropna(),
                        nbinsx=30,
                        name='After',
                        marker_color='green',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title=f'Distribution Comparison: {selected_feature}',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Interactive dashboard not available - processed data not found.")

# Section 10: Key Insights
st.header("10. Key Insights")

st.markdown("""
**Preprocessing Approach:**
- **Data Quality**: Systematic approach to handle missing values and data inconsistencies
- **Feature Integrity**: Feature preservation while optimizing data types
- **Outlier Handling**: Identification and appropriate treatment of outliers
- **Consistency**: Ensuring consistent preprocessing across train/test sets

**Domain Knowledge Integration:**
- Missing values treated based on real estate domain understanding
- Quality ratings handled as ordinal relationships
- Architectural features processed appropriately

**Model Readiness:**
- Missing value treatment enables algorithm training
- Data type optimization improves numerical stability
- Consistent feature sets support robust performance
- Quality corrections prevent erroneous pattern learning

**Next Steps:**
The processed dataset can proceed to feature engineering, where new predictive features
will be created and existing features will be encoded for machine learning algorithms.
""")

# Footer
st.markdown("---")
st.markdown("*This preprocessing pipeline mirrors the systematic approach from the 02_preprocessing.ipynb notebook*")