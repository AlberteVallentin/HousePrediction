import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
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

# Page configuration
st.set_page_config(
    page_title="Data Exploration - Ames Housing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .insight-box {
        background-color: #e8f6f3;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fef9e7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Data Exploration</h1>', unsafe_allow_html=True)
st.markdown("**Exploratory analysis of the Ames Housing dataset to understand structure, identify patterns, and discover data quality issues**")

# Load data with error handling
@st.cache_data
def load_exploration_data():
    """Load data for exploration."""
    try:
        df_combined, df_train, df_test = create_combined_dataset()
        return df_combined, df_train, df_test
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

with st.spinner("Loading dataset..."):
    df_combined, df_train, df_test = load_exploration_data()

if df_combined is None:
    st.error("⚠️ Failed to load data. Please check your data files.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("## Navigation")
sections = {
    "Dataset Overview": "overview",
    "Target Analysis": "target", 
    "Missing Values": "missing",
    "Correlations": "correlations",
    "Categorical Features": "categorical",
    "Numerical Features": "numerical",
    "Neighborhood Insights": "neighborhoods",
    "Data Quality Issues": "quality",
    "Key Findings": "findings"
}

selected_section = st.sidebar.radio("Jump to section:", list(sections.keys()))

# Helper functions for visualizations
@st.cache_data
def create_distribution_plot(data, title, log_scale=False):
    """Create distribution plot with histogram and box plot"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'{title} - Histogram', f'{title} - Box Plot'],
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    plot_data = np.log(data) if log_scale else data
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=plot_data, nbinsx=50, name="Distribution", 
                    marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(x=plot_data, name="Box Plot", marker_color='darkblue'),
        row=2, col=1
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Log Values" if log_scale else "Values")
    
    return fig

@st.cache_data
def create_correlation_plot(df, target_col, top_n=15):
    """Create correlation plot with target variable"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
    
    # Remove the target itself and get top N
    correlations = correlations[1:top_n+1]
    
    fig = go.Figure(data=go.Bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        marker_color=px.colors.sequential.Blues_r
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Features Correlated with {target_col}",
        xaxis_title="Absolute Correlation",
        yaxis_title="Features",
        height=500,
    )
    
    return fig

@st.cache_data
def create_missing_values_plot(df):
    """Create missing values visualization"""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        return None
    
    missing_percent = (missing_data / len(df)) * 100
    
    fig = go.Figure(data=go.Bar(
        x=missing_percent.values,
        y=missing_data.index,
        orientation='h',
        marker_color=px.colors.sequential.Reds_r,
        text=[f'{val:.1f}%' for val in missing_percent.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Missing Values by Feature",
        xaxis_title="Percentage Missing",
        yaxis_title="Features",
        height=max(400, len(missing_data) * 25)
    )
    
    return fig

# SECTION 1: Dataset Overview
if sections[selected_section] == "overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{len(df_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(df_test):,}")
    with col3:
        st.metric("Total Features", f"{df_train.shape[1]}")
    with col4:
        price_range = f"${df_train['SalePrice'].min():,.0f} - ${df_train['SalePrice'].max():,.0f}"
        st.metric("Price Range", price_range)
    
    st.markdown("---")
    
    # Dataset comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Dimensions")
        comparison_data = pd.DataFrame({
            'Dataset': ['Training', 'Test', 'Combined'],
            'Samples': [len(df_train), len(df_test), len(df_combined)],
            'Features': [df_train.shape[1], df_test.shape[1], df_combined.shape[1]-1]
        })
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
    with col2:
        st.subheader("Data Types")
        train_types = df_train.dtypes.value_counts()
        
        # Create pie chart for data types
        fig = go.Figure(data=go.Pie(
            labels=[str(idx) for idx in train_types.index],
            values=train_types.values,
            hole=0.4
        ))
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True, key="data_types_pie")
    
    # Sample data preview
    st.subheader("Data Preview")
    st.dataframe(df_train.head(), use_container_width=True)

# SECTION 2: Target Analysis
elif sections[selected_section] == "target":
    st.markdown('<h2 class="section-header">Target Variable Analysis (SalePrice)</h2>', unsafe_allow_html=True)
    
    if 'SalePrice' in df_train.columns:
        sale_price = df_train['SalePrice']
        
        # Key statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Price", f"${sale_price.mean():,.0f}")
        with col2:
            st.metric("Median Price", f"${sale_price.median():,.0f}")
        with col3:
            st.metric("Skewness", f"{sale_price.skew():.2f}")
        with col4:
            outlier_threshold = sale_price.quantile(0.75) + 1.5 * (sale_price.quantile(0.75) - sale_price.quantile(0.25))
            outliers = len(sale_price[sale_price > outlier_threshold])
            st.metric("Outliers", f"{outliers} ({outliers/len(sale_price)*100:.1f}%)")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Distribution")
            dist_fig = create_distribution_plot(sale_price, "SalePrice")
            st.plotly_chart(dist_fig, use_container_width=True, key="original_distribution")
            
        with col2:
            st.subheader("Log-Transformed Distribution")
            log_dist_fig = create_distribution_plot(sale_price, "Log(SalePrice)", log_scale=True)
            st.plotly_chart(log_dist_fig, use_container_width=True, key="log_distribution")
        
        # Insights
        st.markdown("""
        <div class="insight-box">
        <h4>Key Insights</h4>
        <ul>
            <li><strong>Right-skewed distribution:</strong> Most houses are priced below the mean</li>
            <li><strong>Log transformation:</strong> Improves normality for modeling</li>
            <li><strong>Outliers:</strong> High-end luxury properties require special attention</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# SECTION 3: Missing Values
elif sections[selected_section] == "missing":
    st.markdown('<h2 class="section-header">Missing Values Analysis</h2>', unsafe_allow_html=True)
    
    missing_counts = df_combined.isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_missing = missing_counts.sum()
        st.metric("Total Missing", f"{total_missing:,}")
    with col2:
        missing_pct = (total_missing / (df_combined.shape[0] * df_combined.shape[1])) * 100
        st.metric("Missing %", f"{missing_pct:.2f}%")
    with col3:
        st.metric("Features Affected", f"{len(missing_features)}")
    
    if len(missing_features) > 0:
        # Missing values plot
        missing_plot = create_missing_values_plot(df_combined)
        if missing_plot:
            st.plotly_chart(missing_plot, use_container_width=True, key="missing_values_plot")
        
        # Top missing features table
        st.subheader("Features with Most Missing Values")
        top_missing = missing_features.head(10)
        missing_df = pd.DataFrame({
            'Feature': top_missing.index,
            'Missing Count': top_missing.values,
            'Missing %': (top_missing / len(df_combined) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
        # Pattern analysis
        st.markdown("""
        <div class="warning-box">
        <h4>Missing Value Patterns</h4>
        <ul>
            <li><strong>Pool features (PoolQC):</strong> 99.7% missing - mostly no pools</li>
            <li><strong>Misc features:</strong> 96.4% missing - rare special features</li>
            <li><strong>Alley access:</strong> 93.8% missing - not all houses have alley access</li>
            <li><strong>Garage features:</strong> Some missing where houses have no garage</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("No missing values found in the dataset!")

# SECTION 4: Correlations
elif sections[selected_section] == "correlations":
    st.markdown('<h2 class="section-header">Feature Correlations</h2>', unsafe_allow_html=True)
    
    if 'SalePrice' in df_train.columns:
        # Correlation with target
        corr_plot = create_correlation_plot(df_train, 'SalePrice', top_n=15)
        st.plotly_chart(corr_plot, use_container_width=True, key="correlation_plot")
        
        # High correlations between features
        st.subheader("Highly Correlated Feature Pairs")
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        corr_matrix = df_train[numeric_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': round(corr_val, 3)
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            <div class="warning-box">
            <h4>Multicollinearity Warning</h4>
            <p>These highly correlated features may cause issues in modeling. Consider:</p>
            <ul>
                <li>Feature selection to remove redundant features</li>
                <li>Principal Component Analysis (PCA)</li>
                <li>Regularization techniques (Ridge/Lasso)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# SECTION 5: Categorical Features
elif sections[selected_section] == "categorical":
    st.markdown('<h2 class="section-header">Categorical Features Analysis</h2>', unsafe_allow_html=True)
    
    categorical_cols = df_train.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        # Feature selection
        selected_feature = st.selectbox("Choose a categorical feature to analyze:", categorical_cols)
        
        if selected_feature and 'SalePrice' in df_train.columns:
            # Value counts and price analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_feature} - Distribution")
                value_counts = df_train[selected_feature].value_counts().head(10)
                
                fig = go.Figure(data=go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker_color=px.colors.sequential.Blues_r
                ))
                fig.update_layout(
                    title=f"Top 10 Categories in {selected_feature}",
                    xaxis_title="Category",
                    yaxis_title="Count",
                    height=400
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key=f"categorical_dist_{selected_feature}")
            
            with col2:
                st.subheader(f"{selected_feature} - Price Impact")
                price_by_category = df_train.groupby(selected_feature)['SalePrice'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                
                # Only show categories with sufficient samples
                price_by_category = price_by_category[price_by_category['count'] >= 5].head(10)
                
                if not price_by_category.empty:
                    fig = go.Figure(data=go.Bar(
                        x=price_by_category.index,
                        y=price_by_category['mean'],
                        marker_color=px.colors.sequential.Greens_r,
                        text=[f"${val:,.0f}" for val in price_by_category['mean']],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title=f"Average Price by {selected_feature}",
                        xaxis_title="Category",
                        yaxis_title="Average Price ($)",
                        height=400
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f"categorical_price_{selected_feature}")
            
            # Statistical summary
            st.subheader("Statistical Summary")
            summary_df = price_by_category.round(0)
            summary_df.columns = ['Average Price ($)', 'Sample Count']
            st.dataframe(summary_df, use_container_width=True)

# SECTION 6: Numerical Features  
elif sections[selected_section] == "numerical":
    st.markdown('<h2 class="section-header">Numerical Features Analysis</h2>', unsafe_allow_html=True)
    
    numerical_cols = df_train.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'SalePrice']
    
    if len(numerical_cols) > 0:
        # Feature selection
        selected_feature = st.selectbox("Choose a numerical feature to analyze:", numerical_cols)
        
        if selected_feature and 'SalePrice' in df_train.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_feature} - Distribution")
                dist_fig = create_distribution_plot(df_train[selected_feature], selected_feature)
                st.plotly_chart(dist_fig, use_container_width=True)
            
            with col2:
                st.subheader(f"{selected_feature} vs Price")
                
                # Scatter plot
                fig = go.Figure(data=go.Scatter(
                    x=df_train[selected_feature],
                    y=df_train['SalePrice'],
                    mode='markers',
                    marker=dict(
                        color=df_train['SalePrice'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Sale Price")
                    ),
                    opacity=0.6
                ))
                
                # Add trend line
                z = np.polyfit(df_train[selected_feature].dropna(), 
                              df_train.loc[df_train[selected_feature].notna(), 'SalePrice'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df_train[selected_feature].min(), 
                                    df_train[selected_feature].max(), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_feature} vs Sale Price",
                    xaxis_title=selected_feature,
                    yaxis_title="Sale Price ($)",
                    height=500,
                   
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation coefficient
            correlation = df_train[selected_feature].corr(df_train['SalePrice'])
            if not np.isnan(correlation):
                st.metric("Correlation with Price", f"{correlation:.3f}")

# SECTION 7: Neighborhoods
elif sections[selected_section] == "neighborhoods":
    st.markdown('<h2 class="section-header">Neighborhood Analysis</h2>', unsafe_allow_html=True)
    
    if 'Neighborhood' in df_train.columns and 'SalePrice' in df_train.columns:
        neighborhood_stats = df_train.groupby('Neighborhood').agg({
            'SalePrice': ['mean', 'median', 'count'],
            'GrLivArea': 'mean',
            'OverallQual': 'mean'
        }).round(2)
        
        neighborhood_stats.columns = ['Avg_Price', 'Median_Price', 'Count', 'Avg_Living_Area', 'Avg_Quality']
        neighborhood_stats = neighborhood_stats.sort_values('Avg_Price', ascending=False)
        
        # Top and bottom neighborhoods
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Premium Neighborhoods")
            top_neighborhoods = neighborhood_stats.head(8)
            
            fig = go.Figure(data=go.Bar(
                x=top_neighborhoods['Avg_Price'],
                y=top_neighborhoods.index,
                orientation='h',
                marker_color=px.colors.sequential.Greens_r,
                text=[f"${val:,.0f}" for val in top_neighborhoods['Avg_Price']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Top 8 Neighborhoods by Average Price",
                xaxis_title="Average Price ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key="premium_neighborhoods")
        
        with col2:
            st.subheader("Affordable Neighborhoods")
            bottom_neighborhoods = neighborhood_stats.tail(8)
            
            fig = go.Figure(data=go.Bar(
                x=bottom_neighborhoods['Avg_Price'],
                y=bottom_neighborhoods.index,
                orientation='h',
                marker_color=px.colors.sequential.Blues_r,
                text=[f"${val:,.0f}" for val in bottom_neighborhoods['Avg_Price']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Bottom 8 Neighborhoods by Average Price",
                xaxis_title="Average Price ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True, key="affordable_neighborhoods")
        
        # Detailed neighborhood table
        st.subheader("Neighborhood Statistics")
        display_stats = neighborhood_stats.copy()
        display_stats['Avg_Price'] = display_stats['Avg_Price'].apply(lambda x: f"${x:,.0f}")
        display_stats['Median_Price'] = display_stats['Median_Price'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_stats, use_container_width=True)

# SECTION 8: Data Quality
elif sections[selected_section] == "quality":
    st.markdown('<h2 class="section-header">Data Quality Issues</h2>', unsafe_allow_html=True)
    
    # Load data quality log if available
    try:
        quality_df = load_data_quality_log()
        
        st.subheader("Identified Issues")
        st.markdown("*Issues discovered through systematic validation in notebook 01 and corrected in notebook 02*")
        
        # Issue summary
        issue_counts = quality_df['Issue'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Issues", len(quality_df))
        with col2:
            st.metric("Houses Affected", quality_df['Id'].nunique())
        with col3:
            st.metric("Issue Types", len(issue_counts))
        
        # Issue types distribution
        st.subheader("Issue Types Distribution")
        fig = go.Figure(data=go.Bar(
            x=issue_counts.values,
            y=issue_counts.index,
            orientation='h',
            marker_color=px.colors.sequential.Reds_r,
            text=issue_counts.values,
            textposition='auto'
        ))
        fig.update_layout(
            title="Distribution of Data Quality Issues",
            xaxis_title="Number of Cases",
            yaxis_title="Issue Type",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Houses with multiple issues
        st.subheader("Houses with Multiple Issues")
        multi_issue_houses = quality_df[quality_df['IssueCount'] > 1].groupby('Id').agg({
            'Issue': lambda x: ', '.join(x),
            'IssueCount': 'first'
        }).sort_values('IssueCount', ascending=False)
        
        if not multi_issue_houses.empty:
            st.dataframe(multi_issue_houses, use_container_width=True)
        else:
            st.info("No houses with multiple issues found.")
        
        # Detailed issues table
        st.subheader("Complete Issues Log")
        
        # Add color coding for severity
        def color_severity(val):
            if 'timeline' in val.lower() or 'typo' in val.lower():
                return 'background-color: #ffebee'  # Light red
            elif 'missing' in val.lower():
                return 'background-color: #fff3e0'  # Light orange
            elif 'logic' in val.lower():
                return 'background-color: #fce4ec'  # Light pink
            else:
                return 'background-color: #f3e5f5'  # Light purple
        
        styled_df = quality_df.style.applymap(color_severity, subset=['Issue'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Key insights about data quality
        st.markdown("""
        <div class="warning-box">
        <h4>Data Quality Insights</h4>
        <ul>
            <li><strong>Timeline Issues:</strong> Remodel dates before construction, garage year typos</li>
            <li><strong>Structural Logic:</strong> Pool areas without quality ratings, kitchens with zero count</li>
            <li><strong>Missing Context:</strong> Basement areas without corresponding quality assessments</li>
            <li><strong>Critical Property:</strong> ID 2550 flagged in multiple validation checks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning("Data quality log not found at `data/logs/data_quality_issues.csv`. Issues will be identified during preprocessing.")
        
        # Show expected data quality patterns from the notebooks
        st.subheader("Expected Data Quality Issues")
        st.markdown("""
        Based on the systematic validation approach from notebook 01, typical issues include:
        
        **Timeline Inconsistencies:**
        - Remodel dates before construction dates
        - Garage construction dates with typos (e.g., 2207 instead of 2007)
        
        **Structural Logic Violations:**
        - Pool areas > 0 without pool quality ratings
        - Kitchen counts = 0 but kitchen quality exists
        - Basement areas without basement quality assessments
        
        **Missing Feature Context:**
        - Properties with specific features missing corresponding quality/condition ratings
        - Architectural elements without proper descriptive attributes
        """)
        
        st.info("Run the preprocessing notebook to generate the complete data quality log.")

# SECTION 9: Key Findings
elif sections[selected_section] == "findings":
    st.markdown('<h2 class="section-header">Key Findings</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h3>Target Variable (SalePrice)</h3>
    <ul>
        <li><strong>Distribution:</strong> Right-skewed (skewness = 1.88), requires log transformation</li>
        <li><strong>Range:</strong> $34,900 - $755,000 (mean: $180,921, median: $163,000)</li>
        <li><strong>Outliers:</strong> 61 high-price properties (>$340k) represent 4.2% of data</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h3>Feature Relationships</h3>
    <ul>
        <li><strong>Strongest predictors:</strong> OverallQual (0.79), GrLivArea (0.71), garage features (0.62-0.64)</li>
        <li><strong>Multicollinearity issues:</strong> GarageCars vs GarageArea (0.882), GrLivArea vs TotRmsAbvGrd (0.825)</li>
        <li><strong>Negative correlations:</strong> KitchenAbvGr (-0.14), EnclosedPorch (-0.13)</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h3>Neighborhood Impact</h3>
    <ul>
        <li><strong>Variation:</strong> 25 distinct neighborhoods with significant price differences</li>
        <li><strong>Premium areas:</strong> Can command $100k+ premiums over average</li>
        <li><strong>Location effect:</strong> Can double or halve property values</li>
    </ul>
    </div>
    
    <div class="warning-box">
    <h3>Data Quality Concerns</h3>
    <ul>
        <li><strong>Missing values:</strong> Concentrated in luxury features (pools, misc features)</li>
        <li><strong>Outliers:</strong> Properties 524 and 1299 need special attention</li>
        <li><strong>Type issues:</strong> 3 ordinal features misclassified as numeric</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h3>Modeling Implications</h3>
    <ul>
        <li><strong>Preprocessing:</strong> Log transformation for skewed features recommended</li>
        <li><strong>Feature selection:</strong> Address multicollinearity in garage and area features</li>
        <li><strong>Missing values:</strong> Most are informative (no pool = no pool quality rating)</li>
        <li><strong>Quality over quantity:</strong> Overall quality is more predictive than size alone</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This analysis follows the methodology from notebook 01_data_exploration.ipynb*")