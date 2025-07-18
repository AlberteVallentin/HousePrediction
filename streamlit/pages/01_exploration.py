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
    load_raw_data, 
    create_combined_dataset, 
    load_feature_descriptions_cached,
    load_data_quality_log,
    get_numerical_features,
    get_categorical_features
)
from utils.visualizations import (
    create_missing_values_heatmap,
    create_missing_values_bar_chart,
    create_target_distribution_plot,
    create_correlation_heatmap,
    create_target_correlation_bar_chart,
    create_categorical_boxplot,
    create_scatter_plot,
    create_neighborhood_analysis_plot,
    create_feature_correlation_network,
    create_time_series_analysis,
    create_quality_vs_price_analysis
)

# Page configuration
st.set_page_config(
    page_title="Data Exploration",
    layout="wide"
)

st.title("Data Exploration")
st.markdown("---")

# Load data
@st.cache_data
def load_exploration_data():
    """Load data for exploration."""
    df_combined, df_train, df_test = create_combined_dataset()
    return df_combined, df_train, df_test

df_combined, df_train, df_test = load_exploration_data()

if df_combined is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Section 1: Dataset Overview
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Training Samples", len(df_train))
with col2:
    st.metric("Test Samples", len(df_test))
with col3:
    st.metric("Total Features", df_train.shape[1])

# Dataset shape information
st.subheader("Dataset Dimensions")
overview_data = {
    'Dataset': ['Training', 'Test', 'Combined'],
    'Samples': [len(df_train), len(df_test), len(df_combined)],
    'Features': [df_train.shape[1], df_test.shape[1], df_combined.shape[1]-1]  # -1 for dataset_source
}
st.table(pd.DataFrame(overview_data))

# Data types overview
st.subheader("Data Types Distribution")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Data Types**")
    train_types = df_train.dtypes.value_counts()
    train_types.index = train_types.index.astype(str)
    st.bar_chart(train_types)

with col2:
    st.markdown("**Test Data Types**")
    test_types = df_test.dtypes.value_counts()
    test_types.index = test_types.index.astype(str)
    st.bar_chart(test_types)

# Section 2: Missing Values Analysis
st.header("2. Missing Values Analysis")

# Calculate missing values
missing_counts = df_combined.isnull().sum()
missing_features = missing_counts[missing_counts > 0]

col1, col2, col3 = st.columns(3)
with col1:
    total_missing = missing_counts.sum()
    total_values = df_combined.shape[0] * df_combined.shape[1]
    st.metric("Total Missing Values", f"{total_missing:,}")

with col2:
    missing_pct = (total_missing / total_values) * 100
    st.metric("Missing Percentage", f"{missing_pct:.2f}%")

with col3:
    features_with_missing = len(missing_features)
    st.metric("Features with Missing Values", features_with_missing)

# Missing values visualization
st.subheader("Missing Values by Feature")
missing_bar_chart = create_missing_values_bar_chart(df_combined)
if missing_bar_chart:
    st.plotly_chart(missing_bar_chart, use_container_width=True)

# Top missing features table
st.subheader("Top 15 Features with Missing Values")
if not missing_features.empty:
    top_missing = missing_features.head(15)
    missing_pct = (top_missing / len(df_combined)) * 100
    
    missing_df = pd.DataFrame({
        'Feature': top_missing.index,
        'Missing Count': top_missing.values,
        'Missing Percentage': missing_pct.values
    })
    st.dataframe(missing_df, use_container_width=True)

# Missing values heatmap
st.subheader("Missing Values Pattern")
if "show_missing_heatmap" not in st.session_state:
    st.session_state.show_missing_heatmap = False

if st.button("Show/Hide Missing Values Heatmap"):
    st.session_state.show_missing_heatmap = not st.session_state.show_missing_heatmap

if st.session_state.show_missing_heatmap:
    missing_heatmap = create_missing_values_heatmap(df_combined)
    if missing_heatmap:
        st.plotly_chart(missing_heatmap, use_container_width=True)
    else:
        st.info("No missing values to display")

# Section 3: Target Variable Analysis
st.header("3. Target Variable Analysis (SalePrice)")

if 'SalePrice' in df_train.columns:
    target_col = 'SalePrice'
    
    # Basic statistics
    st.subheader("Basic Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"${df_train[target_col].mean():,.0f}")
    with col2:
        st.metric("Median", f"${df_train[target_col].median():,.0f}")
    with col3:
        st.metric("Min", f"${df_train[target_col].min():,.0f}")
    with col4:
        st.metric("Max", f"${df_train[target_col].max():,.0f}")
    
    # Distribution analysis
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Skewness", f"{df_train[target_col].skew():.3f}")
    with col2:
        st.metric("Kurtosis", f"{df_train[target_col].kurtosis():.3f}")
    
    # Distribution plot
    st.subheader("Distribution Visualization")
    target_dist_plot = create_target_distribution_plot(df_train[target_col])
    st.plotly_chart(target_dist_plot, use_container_width=True)
    
    # Outlier analysis
    st.subheader("Outlier Analysis")
    
    # IQR method
    Q1 = df_train[target_col].quantile(0.25)
    Q3 = df_train[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_train[(df_train[target_col] < lower_bound) | (df_train[target_col] > upper_bound)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower Bound", f"${lower_bound:,.0f}")
    with col2:
        st.metric("Upper Bound", f"${upper_bound:,.0f}")
    with col3:
        st.metric("Outliers", f"{len(outliers)} ({len(outliers)/len(df_train)*100:.1f}%)")

# Section 4: Feature Correlations
st.header("4. Feature Correlations")

# Correlation with target
st.subheader("Feature Correlations with SalePrice")
if 'SalePrice' in df_train.columns:
    target_corr_chart = create_target_correlation_bar_chart(df_train, 'SalePrice')
    st.plotly_chart(target_corr_chart, use_container_width=True)

# Full correlation matrix
st.subheader("Feature Correlation Matrix")
if "show_correlation_heatmap" not in st.session_state:
    st.session_state.show_correlation_heatmap = False

if st.button("Show/Hide Correlation Heatmap"):
    st.session_state.show_correlation_heatmap = not st.session_state.show_correlation_heatmap

if st.session_state.show_correlation_heatmap:
    corr_heatmap = create_correlation_heatmap(df_train)
    st.plotly_chart(corr_heatmap, use_container_width=True)

# Section 5: Categorical Features Analysis
st.header("5. Categorical Features Analysis")

categorical_features = get_categorical_features(df_train)
if categorical_features:
    st.subheader("Categorical Features Overview")
    
    # Feature selection for analysis
    selected_cat_feature = st.selectbox(
        "Select categorical feature to analyze:",
        categorical_features
    )
    
    if selected_cat_feature and 'SalePrice' in df_train.columns:
        # Boxplot
        cat_boxplot = create_categorical_boxplot(df_train, selected_cat_feature, 'SalePrice')
        st.plotly_chart(cat_boxplot, use_container_width=True)
        
        # Summary statistics
        st.subheader(f"Summary Statistics for {selected_cat_feature}")
        category_stats = df_train.groupby(selected_cat_feature)['SalePrice'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        category_stats.columns = ['Count', 'Mean Price', 'Median Price', 'Std Dev']
        st.dataframe(category_stats, use_container_width=True)

# Section 6: Numerical Features Analysis
st.header("6. Numerical Features Analysis")

numerical_features = get_numerical_features(df_train)
if 'SalePrice' in numerical_features:
    numerical_features.remove('SalePrice')

if numerical_features:
    st.subheader("Numerical Features vs SalePrice")
    
    # Feature selection
    selected_num_feature = st.selectbox(
        "Select numerical feature to analyze:",
        numerical_features
    )
    
    if selected_num_feature and 'SalePrice' in df_train.columns:
        # Scatter plot
        scatter_plot = create_scatter_plot(df_train, selected_num_feature, 'SalePrice')
        st.plotly_chart(scatter_plot, use_container_width=True)
        
        # Correlation
        correlation = df_train[selected_num_feature].corr(df_train['SalePrice'])
        st.metric("Correlation with SalePrice", f"{correlation:.3f}")

# Section 7: Data Quality Issues
st.header("7. Data Quality Issues")

# Load data quality log
quality_log = load_data_quality_log()

if not quality_log.empty:
    st.subheader("Data Quality Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Issues", len(quality_log))
    with col2:
        unique_houses = quality_log['Id'].nunique()
        st.metric("Houses Affected", unique_houses)
    
    # Issues by type
    st.subheader("Issues by Type")
    issue_counts = quality_log['Issue'].value_counts()
    st.bar_chart(issue_counts)
    
    # Show detailed log
    if "show_quality_log" not in st.session_state:
        st.session_state.show_quality_log = False

    if st.button("Show/Hide Detailed Quality Log"):
        st.session_state.show_quality_log = not st.session_state.show_quality_log

    if st.session_state.show_quality_log:
        st.dataframe(quality_log, use_container_width=True)
else:
    st.info("No data quality issues log found.")

# Section 8: Neighborhood Analysis
st.header("8. Neighborhood Analysis")

if 'Neighborhood' in df_train.columns and 'SalePrice' in df_train.columns:
    st.markdown("""
    Analyze price patterns across different neighborhoods in Ames, Iowa.
    This comprehensive analysis shows average prices, price variability, and sales volume by neighborhood.
    """)
    
    neighborhood_plot = create_neighborhood_analysis_plot(df_train)
    if neighborhood_plot:
        st.plotly_chart(neighborhood_plot, use_container_width=True)
    
    # Add neighborhood summary statistics
    st.subheader("Top 5 Most Expensive Neighborhoods")
    neighborhood_summary = df_train.groupby('Neighborhood')['SalePrice'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(0)
    neighborhood_summary.columns = ['Houses Sold', 'Average Price', 'Median Price', 'Price Std Dev']
    neighborhood_summary = neighborhood_summary.sort_values('Average Price', ascending=False)
    
    st.dataframe(neighborhood_summary.head(), use_container_width=True)
else:
    st.info("Neighborhood analysis not available - required columns not found.")

# Section 9: Feature Correlation Network
st.header("9. Feature Correlation Network")

if 'SalePrice' in df_train.columns:
    st.markdown("""
    Identify the most important features that correlate with house prices.
    This analysis shows features with correlation coefficient >= 0.3 with SalePrice.
    """)
    
    # Allow user to adjust correlation threshold
    correlation_threshold = st.slider(
        "Correlation Threshold",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="Minimum correlation coefficient to display"
    )
    
    feature_network = create_feature_correlation_network(df_train, 'SalePrice', correlation_threshold)
    if feature_network:
        st.plotly_chart(feature_network, use_container_width=True)
    else:
        st.info(f"No features found with correlation >= {correlation_threshold}")
else:
    st.info("Feature correlation network not available - SalePrice column not found.")

# Section 10: Time Series Analysis
st.header("10. Time Series Analysis")

if 'YearBuilt' in df_train.columns and 'SalePrice' in df_train.columns:
    st.markdown("""
    Analyze how house prices relate to the year they were built.
    This helps identify trends in construction quality and market preferences over time.
    """)
    
    time_series_plot = create_time_series_analysis(df_train)
    if time_series_plot:
        st.plotly_chart(time_series_plot, use_container_width=True)
    
    # Add year-based insights
    st.subheader("Construction Era Analysis")
    df_train['ConstructionEra'] = pd.cut(df_train['YearBuilt'], 
                                       bins=[1800, 1950, 1980, 2000, 2020], 
                                       labels=['Pre-1950', '1950-1980', '1980-2000', '2000+'])
    
    era_analysis = df_train.groupby('ConstructionEra')['SalePrice'].agg([
        'count', 'mean', 'median'
    ]).round(0)
    era_analysis.columns = ['Count', 'Average Price', 'Median Price']
    
    st.dataframe(era_analysis, use_container_width=True)
else:
    st.info("Time series analysis not available - required columns not found.")

# Section 11: Quality vs Price Analysis
st.header("11. Quality vs Price Analysis")

if 'OverallQual' in df_train.columns and 'SalePrice' in df_train.columns:
    st.markdown("""
    Comprehensive analysis of how overall quality ratings impact house prices.
    This multi-dimensional view shows price distributions, averages, and variability by quality level.
    """)
    
    quality_analysis = create_quality_vs_price_analysis(df_train)
    if quality_analysis:
        st.plotly_chart(quality_analysis, use_container_width=True)
    
    # Add quality insights
    st.subheader("Quality Impact Summary")
    quality_summary = df_train.groupby('OverallQual')['SalePrice'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(0)
    quality_summary.columns = ['Count', 'Average Price', 'Median Price', 'Price Std Dev']
    
    # Calculate price increase per quality level
    quality_summary['Price_Increase'] = quality_summary['Average Price'].diff()
    quality_summary['Price_Increase_Pct'] = (quality_summary['Average Price'].pct_change() * 100).round(1)
    
    st.dataframe(quality_summary, use_container_width=True)
    
    # Highlight key insights
    col1, col2 = st.columns(2)
    with col1:
        highest_quality = quality_summary.index.max()
        lowest_quality = quality_summary.index.min()
        price_multiplier = quality_summary.loc[highest_quality, 'Average Price'] / quality_summary.loc[lowest_quality, 'Average Price']
        st.metric("Quality Impact", f"{price_multiplier:.1f}x", 
                 help=f"Houses with quality {highest_quality} cost {price_multiplier:.1f}x more than quality {lowest_quality}")
    
    with col2:
        avg_increase = quality_summary['Price_Increase_Pct'].mean()
        st.metric("Average Price Increase per Quality Level", f"{avg_increase:.1f}%", 
                 help="Average percentage increase in price per quality level improvement")
else:
    st.info("Quality analysis not available - required columns not found.")

# Section 12: Interactive Price Distribution Explorer
st.header("12. Interactive Price Distribution Explorer")

if 'SalePrice' in df_train.columns:
    st.markdown("""
    Explore house price distributions filtered by different characteristics.
    Use the filters below to analyze specific market segments.
    """)
    
    # Create filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Neighborhood' in df_train.columns:
            neighborhoods = ['All'] + sorted(df_train['Neighborhood'].unique().tolist())
            selected_neighborhood = st.selectbox("Filter by Neighborhood", neighborhoods)
        else:
            selected_neighborhood = 'All'
    
    with col2:
        if 'OverallQual' in df_train.columns:
            qualities = ['All'] + sorted(df_train['OverallQual'].unique().tolist())
            selected_quality = st.selectbox("Filter by Quality", qualities)
        else:
            selected_quality = 'All'
    
    with col3:
        price_range = st.slider(
            "Price Range ($)",
            min_value=int(df_train['SalePrice'].min()),
            max_value=int(df_train['SalePrice'].max()),
            value=(int(df_train['SalePrice'].min()), int(df_train['SalePrice'].max())),
            step=10000
        )
    
    # Apply filters
    filtered_df = df_train.copy()
    
    if selected_neighborhood != 'All' and 'Neighborhood' in df_train.columns:
        filtered_df = filtered_df[filtered_df['Neighborhood'] == selected_neighborhood]
    
    if selected_quality != 'All' and 'OverallQual' in df_train.columns:
        filtered_df = filtered_df[filtered_df['OverallQual'] == selected_quality]
    
    filtered_df = filtered_df[
        (filtered_df['SalePrice'] >= price_range[0]) & 
        (filtered_df['SalePrice'] <= price_range[1])
    ]
    
    if len(filtered_df) > 0:
        # Create filtered distribution plot
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered_df['SalePrice'],
            nbinsx=30,
            name='Filtered Distribution',
            marker_color='steelblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Price Distribution (n={len(filtered_df)} houses)',
            xaxis_title='Sale Price ($)',
            yaxis_title='Frequency',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Houses Found", len(filtered_df))
        with col2:
            st.metric("Average Price", f"${filtered_df['SalePrice'].mean():,.0f}")
        with col3:
            st.metric("Median Price", f"${filtered_df['SalePrice'].median():,.0f}")
        with col4:
            st.metric("Price Range", f"${filtered_df['SalePrice'].max() - filtered_df['SalePrice'].min():,.0f}")
    else:
        st.warning("No houses found matching the selected filters.")
else:
    st.info("Interactive price explorer not available - SalePrice column not found.")

# Section 13: Key Insights

st.markdown("""
**Data Quality:**
- Dataset contains 1,460 training samples and 1,459 test samples with no duplicates
- 34 data quality issues identified across 31 houses requiring preprocessing attention
- Critical outliers: Properties 524 and 1299 with luxury features but partial sale prices
- 2 problematic outliers with >4000 sqft living area but <$200k price (likely incomplete sales)

**Target Variable (SalePrice):**
- Strong right-skewed distribution with skewness = 1.88
- Price range: $34,900 - $755,000 (mean: $180,921, median: $163,000)
- Log transformation improves normality and correlation patterns
- 61 high-price outliers (>$340k) represent 4.2% of properties

**Feature Relationships:**
- OverallQual is strongest predictor (correlation: 0.79 with SalePrice)
- GrLivArea (0.71) and garage features (0.62-0.64) are key drivers
- High multicollinearity: GarageCars vs GarageArea (0.882), GrLivArea vs TotRmsAbvGrd (0.825)
- Negative correlations: KitchenAbvGr (-0.14) and EnclosedPorch (-0.13)

**Neighborhood Insights:**
- 25 distinct neighborhoods with significant price variation
- Premium neighborhoods command $100k+ premiums over average
- NAmes is most common neighborhood (443 houses)
- Location effects can double or halve property values

**Quality Impact:**
- Quality ratings show clear step-wise price increases
- Excellent quality homes cost significantly more than average quality
- OverallQual, ExterQual, and KitchenQual are key quality predictors
- Quality beats size as primary value driver

**Preprocessing Implications:**
- 3 misclassified ordinal features need type correction (OverallQual, OverallCond, MSSubClass)
- Missing values concentrated in luxury features (PoolQC 99.7%, MiscFeature 96.4%)
- Log transformation recommended for skewed numerical features
- Feature selection needed for highly correlated pairs to reduce multicollinearity
""")

# Footer
st.markdown("---")
st.markdown("*This analysis mirrors the findings from the 01_exploration.ipynb notebook*")