import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import sys
import os

# Add the parent directory to access utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import (
    load_processed_data,
    load_feature_descriptions_cached,
    get_categorical_features,
    get_numerical_features
)
from utils.visualizations import (
    create_correlation_heatmap,
    create_target_distribution_plot,
    create_feature_importance_plot,
    create_skewness_comparison,
    create_scatter_plot
)

# Page configuration
st.set_page_config(
    page_title="Feature Engineering",
    layout="wide"
)

st.title("Feature Engineering")
st.markdown("---")

st.markdown("""
This page demonstrates the feature engineering process that transforms preprocessed data into 
model-ready features. The process includes creating new features, handling skewness, and encoding 
categorical variables.
""")

# Load data
@st.cache_data
def load_feature_engineering_data():
    """Load processed data for feature engineering analysis."""
    processed_data = load_processed_data()
    feature_descriptions = load_feature_descriptions_cached()
    return processed_data, feature_descriptions

processed_data, feature_descriptions = load_feature_engineering_data()

if not processed_data or 'train_cleaned' not in processed_data:
    st.error("Processed data not found. Please run the preprocessing pipeline first.")
    st.stop()

# Load the cleaned data
df_train = processed_data['train_cleaned']
df_test = processed_data['test_cleaned']

# Section 1: Initial Data Overview
st.header("1. Initial Data Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Samples", len(df_train))
with col2:
    st.metric("Test Samples", len(df_test))
with col3:
    st.metric("Initial Features", df_train.shape[1] - 1)  # -1 for SalePrice

# Section 2: Feature Creation
st.header("2. Feature Creation")

st.markdown("""
New features are created based on domain knowledge and data analysis:
""")

# Feature creation examples
feature_creation_examples = {
    'New Feature': [
        'GarageAge',
        'HouseAge', 
        'YearsSinceRemodel',
        'HasGarage',
        'BsmtFinSF',
        'TotalFlrSF',
        'TotalBaths',
        'GarageAreaPerCar'
    ],
    'Formula': [
        'YrSold - GarageYrBlt',
        'YrSold - YearBuilt',
        'YrSold - YearRemodAdd',
        'Binary indicator for garage presence',
        'BsmtFinSF1 + BsmtFinSF2',
        '1stFlrSF + 2ndFlrSF',
        'FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath',
        'GarageArea / GarageCars'
    ],
    'Purpose': [
        'Capture garage depreciation over time',
        'Capture house depreciation over time',
        'Measure recency of renovations',
        'Distinguish houses with/without garage',
        'Consolidate finished basement area',
        'Total above-ground living space',
        'Weighted total bathroom count',
        'Garage space efficiency per car'
    ]
}

feature_creation_df = pd.DataFrame(feature_creation_examples)
st.dataframe(feature_creation_df, use_container_width=True)

# Section 3: Correlation Analysis
st.header("3. Correlation Analysis")

st.markdown("""
Analysis of feature correlations revealed highly correlated pairs that needed consolidation:
""")

# Show correlation findings
correlation_findings = {
    'Feature Pair': [
        'GarageArea & GarageCars',
        'TotRmsAbvGrd & GrLivArea',
        '1stFlrSF & TotalBsmtSF'
    ],
    'Correlation': [0.89, 0.81, 0.79],
    'Resolution': [
        'Created GarageAreaPerCar ratio',
        'Kept both as they measure different aspects',
        'Consolidated into TotalFlrSF'
    ]
}

correlation_df = pd.DataFrame(correlation_findings)
st.dataframe(correlation_df, use_container_width=True)

# Section 4: Skewness Analysis and Transformation
st.header("4. Skewness Analysis and Transformation")

st.markdown("""
Numerical features were analyzed for skewness and transformed using log1p where necessary:
""")

# Show skewness transformation results
skewness_data = {
    'Metric': [
        'Features with |skew| ≥ 0.5 (before)',
        'Features with |skew| ≥ 0.5 (after)',
        'Average skewness (before)',
        'Average skewness (after)'
    ],
    'Value': [
        '23 features',
        '16 features',
        '4.07',
        '2.28'
    ],
    'Improvement': [
        '-7 features',
        '-',
        '-1.79',
        '-'
    ]
}

skewness_df = pd.DataFrame(skewness_data)
st.dataframe(skewness_df, use_container_width=True)

# Interactive Skewness Transformation Analysis
st.subheader("Interactive Skewness Transformation Analysis")

st.markdown("""
Explore the effect of log transformation on different features.
Select a feature to see how log transformation improves its distribution.
""")

# List of features that actually underwent log transformation (skewed features from notebook)
log_transform_candidates = ['LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtUnfSF', 
                           'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 
                           'ScreenPorch', 'PoolArea', 'MiscVal', 'GarageAge', 'HasGarage', 
                           'HouseAge', 'BsmtFinSF']

if df_train is not None and len(df_train) > 0:
    # Filter available features
    available_features = [col for col in log_transform_candidates if col in df_train.columns]
    
    if available_features:
        selected_feature = st.selectbox(
            "Select feature for skewness analysis:",
            available_features
        )
        
        # Get original data from cleaned dataset (before skewness transformation)
        original_data = df_train[selected_feature].dropna()
        
        # Apply log transformation (same as in notebook)
        transformed_data = np.log1p(original_data)
        
        # Create skewness comparison
        skewness_plot = create_skewness_comparison(original_data, transformed_data, selected_feature)
        st.plotly_chart(skewness_plot, use_container_width=True)
        
        # Show improvement metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            orig_skew = stats.skew(original_data)
            st.metric("Original Skewness", f"{orig_skew:.3f}")
        with col2:
            trans_skew = stats.skew(transformed_data)
            st.metric("Transformed Skewness", f"{trans_skew:.3f}")
        with col3:
            improvement = abs(orig_skew) - abs(trans_skew)
            st.metric("Improvement", f"{improvement:.3f}", 
                     delta=f"{improvement:.3f}" if improvement > 0 else None)
    else:
        st.info("No suitable features found for skewness analysis.")
else:
    st.info("Skewness analysis requires the actual dataset to be loaded.")

# Section 5: Categorical Feature Encoding
st.header("5. Categorical Feature Encoding")

st.markdown("""
Categorical features were encoded using two strategies based on their nature:
""")

# Encoding strategy breakdown
encoding_strategies = {
    'Encoding Type': ['Label Encoding', 'One-Hot Encoding'],
    'Feature Count': ['22 features', '21 features'],
    'Purpose': [
        'Preserve ordinal relationships in quality/condition ratings',
        'Handle nominal categorical variables without order'
    ],
    'Examples': [
        'ExterQual (Po→Fa→TA→Gd→Ex), LotShape (IR3→IR2→IR1→Reg)',
        'Neighborhood, SaleType, Foundation, RoofMatl'
    ]
}

encoding_df = pd.DataFrame(encoding_strategies)
st.dataframe(encoding_df, use_container_width=True)

# Show specific label encoding examples
st.subheader("Label Encoding Examples")

label_encoding_examples = {
    'Feature': [
        'ExterQual',
        'LotShape', 
        'BsmtFinType1',
        'Functional'
    ],
    'Mapping': [
        'None:0, Po:1, Fa:2, TA:3, Gd:4, Ex:5',
        'None:0, IR3:1, IR2:2, IR1:3, Reg:4',
        'None:0, Unf:1, LwQ:2, Rec:3, BLQ:4, ALQ:5, GLQ:6',
        'None:0, Sal:1, Sev:2, Maj2:3, Maj1:4, Mod:5, Min2:6, Min1:7, Typ:8'
    ]
}

label_encoding_df = pd.DataFrame(label_encoding_examples)
st.dataframe(label_encoding_df, use_container_width=True)

# Section 6: Target Variable Transformation
st.header("6. Target Variable Transformation")

st.markdown("""
The target variable (SalePrice) was highly skewed and required transformation:
""")

# Show target transformation metrics
target_transformation = {
    'Metric': [
        'Original Skewness',
        'Transformed Skewness',
        'Improvement'
    ],
    'Value': [
        '1.879',
        '0.121',
        '1.758'
    ]
}

target_df = pd.DataFrame(target_transformation)
st.dataframe(target_df, use_container_width=True)

# Create demonstration of target transformation
if 'SalePrice' in df_train.columns:
    st.subheader("Target Variable Distribution")
    
    # Create target distribution plot
    target_plot = create_target_distribution_plot(df_train['SalePrice'])
    st.plotly_chart(target_plot, use_container_width=True)

# Section 7: Feature Engineering Results
st.header("7. Feature Engineering Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Before Feature Engineering")
    st.metric("Features", "80")
    st.metric("Categorical Features", "43")
    st.metric("Numerical Features", "37")
    st.metric("Skewed Features", "23")

with col2:
    st.subheader("After Feature Engineering")
    st.metric("Features", "191")
    st.metric("Encoded Features", "191")
    st.metric("New Features Created", "8")
    st.metric("Skewed Features", "16")

# Section 8: Final Feature Set
st.header("8. Final Feature Set")

st.markdown("""
The feature engineering process resulted in a comprehensive set of 191 features:
""")

# Show final feature breakdown
final_features = {
    'Feature Type': [
        'Label Encoded (Ordinal)',
        'One-Hot Encoded (Nominal)',
        'Numerical (Original)',
        'Numerical (Engineered)',
        'Transformed (Log)',
        'Binary Indicators'
    ],
    'Count': [
        '22',
        '149',
        '20',
        '8',
        '23',
        '3'
    ],
    'Examples': [
        'ExterQual, LotShape, BsmtFinType1',
        'Neighborhood_*, SaleType_*, Foundation_*',
        'OverallQual, YearsSinceRemodel, TotalBaths',
        'HouseAge, GarageAge, TotalFlrSF',
        'LotArea, GrLivArea, MasVnrArea',
        'HasGarage, CentralAir, PavedDrive'
    ]
}

final_features_df = pd.DataFrame(final_features)
st.dataframe(final_features_df, use_container_width=True)

# Show sample of final features if available
if processed_data and 'X_train_final' in processed_data:
    st.subheader("Sample of Final Feature Set")
    final_sample = processed_data['X_train_final'].head(5)
    st.dataframe(final_sample, use_container_width=True)

# Section 9: Quality Validation
st.header("9. Quality Validation")

st.markdown("""
**Final Feature Set Validation:**
- **No Missing Values**: All features have complete data
- **No Infinite Values**: All transformations handled edge cases
- **Proper Encoding**: Categorical variables appropriately encoded
- **Reduced Skewness**: 23 → 16 highly skewed features
- **Domain Knowledge**: New features based on real estate expertise
- **Consistent Scaling**: Train/test consistency maintained
""")

# Section 10: Feature Engineering Impact Analysis
st.header("10. Feature Engineering Impact Analysis")

st.markdown("""
Analyze the impact of engineered features on model performance.
This section shows how new features contribute to predictive power.
""")

# Create tabs for different impact analyses
tab1, tab2, tab3 = st.tabs(["New Feature Analysis", "Correlation Changes", "Feature Importance Simulation"])

with tab1:
    st.subheader("New Feature Effectiveness")
    
    # Show engineered features and their formulas
    engineered_features = {
        'HouseAge': 'YearBuilt → Current Year',
        'YearsSinceRemodel': 'YearRemodAdd → Current Year', 
        'TotalFlrSF': '1stFlrSF + 2ndFlrSF',
        'TotalBaths': 'FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath',
        'BsmtFinSF': 'BsmtFinSF1 + BsmtFinSF2',
        'GarageAreaPerCar': 'GarageArea / GarageCars',
        'HasGarage': 'Binary indicator for garage presence',
        'GarageAge': 'YearBuilt - GarageYrBlt'
    }
    
    # Create feature summary table
    feature_summary = pd.DataFrame({
        'Engineered Feature': list(engineered_features.keys()),
        'Formula': list(engineered_features.values())
    })
    
    st.markdown("**Engineered Features Overview:**")
    st.dataframe(feature_summary, use_container_width=True)
    
    st.markdown("**Engineered Feature Impact Analysis:**")
    st.write("• **Age Features**: HouseAge and GarageAge capture depreciation patterns")
    st.write("• **Consolidation Features**: TotalFlrSF and BsmtFinSF reduce multicollinearity")
    st.write("• **Ratio Features**: GarageAreaPerCar provides efficiency metrics")
    st.write("• **Derived Features**: TotalBaths weighted calculation improves bathroom representation")
    st.write("• **Binary Indicators**: HasGarage creates clear presence/absence signals")
    
    st.info("💡 **Key Insight**: Engineered features based on domain knowledge often outperform raw measurements in predictive power.")

with tab2:
    st.subheader("Correlation Network Changes")
    
    st.markdown("**Major Correlation Changes After Feature Engineering:**")
    st.write("• **Multicollinearity Reduction**: GarageCars vs GarageArea (0.882) → GarageAreaPerCar ratio")
    st.write("• **Feature Consolidation**: TotalFlrSF combines 1stFlrSF + 2ndFlrSF correlation patterns")
    st.write("• **Age Relationships**: HouseAge and GarageAge create temporal correlation structure")
    st.write("• **Bathroom Optimization**: TotalBaths weighted metric improves target correlation")
    
    st.markdown("**Key Improvements:**")
    st.write("• Reduced redundancy in highly correlated feature pairs")
    st.write("• Enhanced predictive power through domain knowledge integration")
    st.write("• Improved model stability by addressing multicollinearity")
    st.write("• Better feature interpretability for business understanding")

with tab3:
    st.subheader("Feature Importance Analysis")
    
    st.markdown("**Expected Feature Importance Patterns:**")
    st.write("• **Quality Features**: OverallQual, ExterQual, KitchenQual likely top predictors")
    st.write("• **Size Features**: GrLivArea, TotalFlrSF expected high importance")
    st.write("• **Age Features**: HouseAge, YearsSinceRemodel capture temporal effects")
    st.write("• **Location Features**: Neighborhood one-hot encoded features provide market segmentation")
    st.write("• **Engineered Features**: TotalBaths, GarageAreaPerCar should outperform raw components")
    
    st.markdown("**Engineering Success Indicators:**")
    st.write("• Engineered features rank higher than their raw components")
    st.write("• Reduced importance of highly correlated original features")
    st.write("• Age-based features capture market depreciation patterns")
    st.write("• Domain-specific ratios provide better signal than raw areas")

# Section 11: Feature Engineering Validation
st.header("11. Feature Engineering Validation")

st.markdown("**Feature Engineering Validation Results:**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Quantitative Improvements:**")
    st.write("• **Feature Count**: 80 → 191 features (+111)")
    st.write("• **Skewness Reduction**: 23 → 16 highly skewed features")
    st.write("• **Multicollinearity**: Reduced through ratio and consolidation features")
    st.write("• **Encoding Completeness**: 100% categorical feature encoding")

with col2:
    st.markdown("**Qualitative Improvements:**")
    st.write("• **Domain Knowledge**: Real estate expertise integrated")
    st.write("• **Interpretability**: Business-meaningful engineered features")
    st.write("• **Model Readiness**: All features properly scaled and encoded")
    st.write("• **Target Alignment**: Log transformation improves linear relationships")

# Section 12: Key Insights
st.header("12. Key Insights")

st.markdown("""
**Feature Engineering Approach:**
- **Feature Expansion**: Features increased through encoding and engineering
- **Skewness Reduction**: Improvements in data distribution through transformations
- **Multicollinearity**: High correlation features consolidated appropriately
- **Domain Knowledge**: Age-based and ratio features capture property characteristics

**Encoding Strategy:**
- **Ordinal Preservation**: Quality ratings maintain natural hierarchy
- **Nominal Handling**: Geographic and categorical variables properly encoded
- **Efficiency**: Balanced approach between interpretability and performance

**Model Readiness:**
- **Numerical Stability**: Log transformations help with extreme values
- **Feature Richness**: Comprehensive property representation through features
- **Consistency**: Train/test feature sets maintained consistently
- **Target Optimization**: Target variable processing improves model performance

**Real Estate Insights:**
- **Age Features**: House and garage age features capture depreciation patterns
- **Quality Hierarchy**: Ordinal encoding preserves condition rating relationships
- **Space Efficiency**: Ratio features provide efficiency measurements
- **Completeness**: Combined features provide holistic property views

**Next Steps:**
The engineered feature set can proceed to model training, where various algorithms
will be tested and compared to find the best performing model for house price prediction.
""")

# Footer
st.markdown("---")
st.markdown("*This feature engineering pipeline mirrors the comprehensive approach from the 03_feature_engineering.ipynb notebook*")