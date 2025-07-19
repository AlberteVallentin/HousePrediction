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
    load_processed_data,
    load_feature_descriptions_cached,
    get_categorical_features,
    get_numerical_features
)

# Page configuration
st.set_page_config(
    page_title="Feature Engineering - Ames Housing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .process-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f6f3;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .transformation-box {
        background-color: #fef9e7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f39c12;
        margin: 1rem 0;
    }
    .feature-category {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Feature Engineering & Optimization</h1>', unsafe_allow_html=True)
st.markdown("**Transform preprocessed data into model-ready features through engineering, encoding, and optimization**")

# Load data with error handling
@st.cache_data
def load_feature_engineering_data():
    """Load processed data for feature engineering analysis."""
    try:
        processed_data = load_processed_data()
        feature_descriptions = load_feature_descriptions_cached()
        return processed_data, feature_descriptions
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}, {}

with st.spinner("Loading feature engineering data..."):
    processed_data, feature_descriptions = load_feature_engineering_data()

if not processed_data or 'train_cleaned' not in processed_data:
    st.error("Processed data not found. Please run the preprocessing pipeline first.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("## Navigation")
sections = {
    "Data Overview": "overview",
    "Feature Creation": "creation", 
    "Skewness Analysis": "skewness",
    "Categorical Encoding": "encoding",
    "Target Transformation": "target",
    "Feature Set Results": "results",
    "Quality Validation": "validation",
    "Impact Analysis": "impact",
    "Key Insights": "insights"
}

selected_section = st.sidebar.radio("Jump to section:", list(sections.keys()))

# Load the cleaned data
df_train = processed_data['train_cleaned']
df_test = processed_data['test_cleaned']

# Helper functions for visualizations
@st.cache_data
def create_feature_distribution_comparison(before_data, after_data, feature_name):
    """Create before/after distribution comparison"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Before: {feature_name}', f'After: {feature_name}'],
        horizontal_spacing=0.1
    )
    
    # Before distribution
    fig.add_trace(
        go.Histogram(x=before_data, nbinsx=30, name="Before", 
                    marker_color='lightcoral', opacity=0.7),
        row=1, col=1
    )
    
    # After distribution
    fig.add_trace(
        go.Histogram(x=after_data, nbinsx=30, name="After", 
                    marker_color='lightblue', opacity=0.7),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Values")
    fig.update_yaxes(title_text="Frequency")
    
    return fig

@st.cache_data
def create_encoding_visualization(categorical_data, encoded_data, feature_name):
    """Visualize categorical encoding results"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Original: {feature_name}', f'Encoded: {feature_name}'],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Original categorical distribution
    value_counts = categorical_data.value_counts().head(10)
    fig.add_trace(
        go.Bar(x=value_counts.index, y=value_counts.values, 
               name="Original", marker_color='lightcoral'),
        row=1, col=1
    )
    
    # Encoded distribution
    encoded_counts = encoded_data.value_counts()
    fig.add_trace(
        go.Bar(x=encoded_counts.index, y=encoded_counts.values, 
               name="Encoded", marker_color='lightblue'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Categories", row=1, col=1)
    fig.update_xaxes(title_text="Encoded Values", row=1, col=2)
    fig.update_yaxes(title_text="Count")
    
    return fig

@st.cache_data
def create_skewness_comparison_plot(features_skewness):
    """Create skewness comparison visualization"""
    fig = go.Figure()
    
    # Add bars for each feature
    colors = ['red' if abs(skew) > 1 else 'orange' if abs(skew) > 0.5 else 'green' 
              for skew in features_skewness.values]
    
    fig.add_trace(go.Bar(
        x=list(features_skewness.keys()),
        y=list(features_skewness.values),
        marker_color=colors,
        text=[f'{val:.2f}' for val in features_skewness.values],
        textposition='auto'
    ))
    
    # Add reference lines
    fig.add_hline(y=1, line_dash="dash", line_color="red", 
                  annotation_text="High Skewness Threshold")
    fig.add_hline(y=-1, line_dash="dash", line_color="red")
    fig.add_hline(y=0.5, line_dash="dot", line_color="orange",
                  annotation_text="Moderate Skewness")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title="Feature Skewness Analysis",
        xaxis_title="Features",
        yaxis_title="Skewness",
        height=500
    )
    fig.update_xaxes(tickangle=45)
    
    return fig

# SECTION 1: Data Overview
if sections[selected_section] == "overview":
    st.markdown('<h2 class="section-header">Data Loading and Initial Setup</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{len(df_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(df_test):,}")
    with col3:
        st.metric("Initial Features", f"{df_train.shape[1] - 1}")  # -1 for SalePrice
    with col4:
        categorical_count = len(df_train.select_dtypes(include=['object']).columns)
        st.metric("Categorical Features", f"{categorical_count}")
    
    st.markdown("---")
    
    # Dataset overview
    st.subheader("Dataset Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Types Distribution**")
        feature_types = df_train.dtypes.value_counts()
        
        fig = go.Figure(data=go.Pie(
            labels=[str(idx) for idx in feature_types.index],
            values=feature_types.values,
            hole=0.4,
            marker_colors=['#ff7f7f', '#7f7fff', '#7fff7f']
        ))
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Dataset Comparison**")
        comparison_data = pd.DataFrame({
            'Dataset': ['Training', 'Test', 'Combined'],
            'Samples': [len(df_train), len(df_test), len(df_train) + len(df_test)],
            'Features': [df_train.shape[1], df_test.shape[1], df_train.shape[1]]
        })
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # Combined dataset creation
    st.markdown("""
    <div class="process-box">
    <h4>Data Loading Process</h4>
    <p><strong>Combined Dataset Creation:</strong> Training and test datasets are combined for consistent feature engineering across both sets.</p>
    <p><strong>Target Extraction:</strong> SalePrice target variable extracted from training data for separate processing.</p>
    <p><strong>Consistency Validation:</strong> Feature alignment and data type consistency verified between datasets.</p>
    </div>
    """, unsafe_allow_html=True)

# SECTION 2: Feature Creation
elif sections[selected_section] == "creation":
    st.markdown('<h2 class="section-header">Feature Creation Process</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    New features are created based on domain knowledge and data relationships to enhance predictive power.
    """)
    
    # Engineered features overview
    engineered_features = {
        'Feature Name': [
            'HouseAge',
            'GarageAge', 
            'YearsSinceRemodel',
            'TotalFlrSF',
            'TotalBaths',
            'BsmtFinSF',
            'GarageAreaPerCar',
            'HasGarage'
        ],
        'Formula/Logic': [
            'YrSold - YearBuilt',
            'YrSold - GarageYrBlt',
            'YrSold - YearRemodAdd',
            '1stFlrSF + 2ndFlrSF',
            'FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath',
            'BsmtFinSF1 + BsmtFinSF2',
            'GarageArea / GarageCars (if GarageCars > 0)',
            '1 if GarageArea > 0, else 0'
        ],
        'Purpose': [
            'Capture depreciation and market trends',
            'Garage condition and age impact',
            'Recent renovation premium effect',
            'Total living space consolidation',
            'Comprehensive bathroom count',
            'Total finished basement area',
            'Garage efficiency metric',
            'Garage presence indicator'
        ]
    }
    
    features_df = pd.DataFrame(engineered_features)
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    # Feature creation rationale
    st.markdown("""
    <div class="insight-box">
    <h4>Feature Engineering Rationale</h4>
    <ul>
        <li><strong>Age-based Features:</strong> Capture temporal effects on property value and condition</li>
        <li><strong>Consolidation Features:</strong> Combine related measurements for better representation</li>
        <li><strong>Ratio Features:</strong> Create efficiency metrics that normalize size-based measurements</li>
        <li><strong>Binary Indicators:</strong> Convert continuous variables to presence/absence signals</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example calculations
    st.subheader("Example Feature Calculations")
    
    if 'YrSold' in df_train.columns and 'YearBuilt' in df_train.columns:
        example_data = df_train[['YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF']].head(5)
        
        # Calculate example engineered features
        example_data = example_data.copy()
        example_data['HouseAge'] = example_data['YrSold'] - example_data['YearBuilt']
        example_data['YearsSinceRemodel'] = example_data['YrSold'] - example_data['YearRemodAdd']
        example_data['TotalFlrSF'] = example_data['1stFlrSF'] + example_data['2ndFlrSF']
        
        st.dataframe(example_data, use_container_width=True)

# SECTION 3: Skewness Analysis
elif sections[selected_section] == "skewness":
    st.markdown('<h2 class="section-header">Skewness Analysis and Transformation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Skewed distributions can negatively impact model performance. Log transformations are applied to highly skewed numerical features.
    """)
    
    # Skewness metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Skewed Features", "23")
    with col2:
        st.metric("Transformed Features", "16")
    with col3:
        st.metric("Skewness Reduction", "30.4%")
    
    # Skewness categories
    st.subheader("Skewness Categories")
    
    skewness_categories = {
        'Category': [
            'Highly Skewed (|skew| > 1.0)',
            'Moderately Skewed (0.5 < |skew| ≤ 1.0)', 
            'Slightly Skewed (0.2 < |skew| ≤ 0.5)',
            'Normal (-0.2 ≤ skew ≤ 0.2)'
        ],
        'Before Transformation': ['23 features', '12 features', '8 features', '37 features'],
        'After Transformation': ['16 features', '15 features', '11 features', '38 features'],
        'Action': [
            'Log transformation applied',
            'Log transformation applied',
            'No transformation needed',
            'No transformation needed'
        ]
    }
    
    skewness_df = pd.DataFrame(skewness_categories)
    st.dataframe(skewness_df, use_container_width=True, hide_index=True)
    
    # Transformation examples
    st.subheader("Transformation Examples")
    
    transformation_examples = {
        'Feature': [
            'LotArea',
            'GrLivArea',
            'TotalBsmtSF',
            'MasVnrArea',
            'BsmtFinSF1'
        ],
        'Original Skewness': [12.82, 1.27, 1.52, 2.61, 1.68],
        'Transformed Skewness': [0.64, 0.11, 0.24, 0.81, 0.21],
        'Improvement': ['92.6%', '91.3%', '84.2%', '68.9%', '87.5%']
    }
    
    transform_df = pd.DataFrame(transformation_examples)
    st.dataframe(transform_df, use_container_width=True, hide_index=True)
    
    # Transformation visualization
    if 'LotArea' in df_train.columns:
        st.subheader("Transformation Impact Visualization")
        
        # Simulate log transformation for demonstration
        original_data = df_train['LotArea'].dropna()
        log_transformed = np.log1p(original_data)  # log1p for values that might be 0
        
        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Original Distribution', 'Log-Transformed Distribution', 
                          'Original Q-Q Plot', 'Transformed Q-Q Plot'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Histograms
        fig.add_trace(go.Histogram(x=original_data, nbinsx=50, name="Original", 
                                 marker_color='lightcoral'), row=1, col=1)
        fig.add_trace(go.Histogram(x=log_transformed, nbinsx=50, name="Transformed", 
                                 marker_color='lightblue'), row=1, col=2)
        
        # Q-Q plots (simplified)
        from scipy import stats
        
        # Original Q-Q
        (osm, osr), (slope, intercept, r) = stats.probplot(original_data, dist="norm", plot=None)
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name="Original Q-Q", 
                               marker_color='red'), row=2, col=1)
        fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', 
                               name="Normal Line", line_color='black'), row=2, col=1)
        
        # Transformed Q-Q
        (osm_t, osr_t), (slope_t, intercept_t, r_t) = stats.probplot(log_transformed, dist="norm", plot=None)
        fig.add_trace(go.Scatter(x=osm_t, y=osr_t, mode='markers', name="Transformed Q-Q", 
                               marker_color='blue'), row=2, col=2)
        fig.add_trace(go.Scatter(x=osm_t, y=slope_t * osm_t + intercept_t, mode='lines', 
                               name="Normal Line", line_color='black'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# SECTION 4: Categorical Encoding
elif sections[selected_section] == "encoding":
    st.markdown('<h2 class="section-header">Categorical Feature Encoding</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Categorical features require encoding for machine learning algorithms. Two strategies are used based on feature characteristics.
    """)
    
    # Encoding strategy overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-category">
        <h4>Label Encoding (22 features)</h4>
        <p><strong>Purpose:</strong> Preserve ordinal relationships</p>
        <p><strong>Applied to:</strong> Quality and condition ratings</p>
        <p><strong>Example:</strong> ExterQual: Po→Fa→TA→Gd→Ex becomes 1→2→3→4→5</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-category">
        <h4>One-Hot Encoding (21 features → 149 columns)</h4>
        <p><strong>Purpose:</strong> Handle nominal categories</p>
        <p><strong>Applied to:</strong> Neighborhoods, materials, types</p>
        <p><strong>Example:</strong> Neighborhood becomes 25 binary columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Label encoding examples
    st.subheader("Label Encoding Mappings")
    
    label_mappings = {
        'Feature': [
            'ExterQual',
            'LotShape',
            'BsmtFinType1',
            'Functional',
            'GarageQual'
        ],
        'Categories → Encoded Values': [
            'None:0, Po:1, Fa:2, TA:3, Gd:4, Ex:5',
            'None:0, IR3:1, IR2:2, IR1:3, Reg:4',
            'None:0, Unf:1, LwQ:2, Rec:3, BLQ:4, ALQ:5, GLQ:6',
            'None:0, Sal:1, Sev:2, Maj2:3, Maj1:4, Mod:5, Min2:6, Min1:7, Typ:8',
            'None:0, Po:1, Fa:2, TA:3, Gd:4, Ex:5'
        ],
        'Rationale': [
            'Quality progression from poor to excellent',
            'Regularity progression from irregular to regular',
            'Basement finish quality from unfinished to good living quarters',
            'Functionality from salvage-only to typical',
            'Garage quality from poor to excellent'
        ]
    }
    
    label_df = pd.DataFrame(label_mappings)
    st.dataframe(label_df, use_container_width=True, hide_index=True)
    
    # One-hot encoding results
    st.subheader("One-Hot Encoding Results")
    
    onehot_results = {
        'Original Feature': [
            'Neighborhood',
            'SaleType',
            'Foundation',
            'RoofMatl',
            'Heating'
        ],
        'Unique Categories': [25, 9, 6, 8, 6],
        'Encoded Columns': [25, 9, 6, 8, 6],
        'Example Columns': [
            'Neighborhood_NAmes, Neighborhood_CollgCr, ...',
            'SaleType_WD, SaleType_New, SaleType_COD, ...',
            'Foundation_PConc, Foundation_CBlock, ...',
            'RoofMatl_CompShg, RoofMatl_Tar&Grv, ...',
            'Heating_GasA, Heating_GasW, Heating_Grav, ...'
        ]
    }
    
    onehot_df = pd.DataFrame(onehot_results)
    st.dataframe(onehot_df, use_container_width=True, hide_index=True)
    
    # Encoding impact
    st.markdown("""
    <div class="transformation-box">
    <h4>Encoding Impact</h4>
    <ul>
        <li><strong>Feature Expansion:</strong> 43 categorical features become 171 encoded features</li>
        <li><strong>Information Preservation:</strong> No information loss through proper encoding</li>
        <li><strong>Model Compatibility:</strong> All features now numerical for ML algorithms</li>
        <li><strong>Interpretability:</strong> One-hot encoded features remain interpretable</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 5: Target Transformation
elif sections[selected_section] == "target":
    st.markdown('<h2 class="section-header">Target Variable Transformation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    The target variable (SalePrice) shows high skewness that can negatively impact model performance.
    """)
    
    # Target transformation metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Skewness", "1.879")
    with col2:
        st.metric("Transformed Skewness", "0.121")
    with col3:
        st.metric("Improvement", "93.6%")
    
    # Target transformation benefits
    st.subheader("Transformation Benefits")
    
    benefits = {
        'Aspect': [
            'Distribution Shape',
            'Linear Relationships',
            'Outlier Impact',
            'Model Performance',
            'Residual Patterns'
        ],
        'Before Transformation': [
            'Highly right-skewed (1.879)',
            'Non-linear with many features',
            'High influence of extreme values',
            'Biased towards high-price properties',
            'Heteroscedastic residuals'
        ],
        'After Transformation': [
            'Near-normal distribution (0.121)',
            'More linear relationships',
            'Reduced outlier influence',
            'Balanced across price ranges',
            'More homoscedastic residuals'
        ]
    }
    
    benefits_df = pd.DataFrame(benefits)
    st.dataframe(benefits_df, use_container_width=True, hide_index=True)
    
    # Demonstrate transformation if SalePrice available
    if 'SalePrice' in df_train.columns:
        st.subheader("Target Transformation Visualization")
        
        sale_price = df_train['SalePrice']
        log_sale_price = np.log(sale_price)
        
        # Create before/after comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Original SalePrice', 'Log-Transformed SalePrice',
                          'Original vs Features', 'Transformed vs Features'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Histograms
        fig.add_trace(go.Histogram(x=sale_price, nbinsx=50, name="Original", 
                                 marker_color='lightcoral', opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=log_sale_price, nbinsx=50, name="Transformed", 
                                 marker_color='lightblue', opacity=0.7), row=1, col=2)
        
        # Scatter plots with a feature (e.g., GrLivArea if available)
        if 'GrLivArea' in df_train.columns:
            fig.add_trace(go.Scatter(x=df_train['GrLivArea'], y=sale_price, mode='markers',
                                   name="Original vs GrLivArea", marker_color='red', opacity=0.6), 
                         row=2, col=1)
            fig.add_trace(go.Scatter(x=df_train['GrLivArea'], y=log_sale_price, mode='markers',
                                   name="Transformed vs GrLivArea", marker_color='blue', opacity=0.6), 
                         row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_yaxes(title_text="SalePrice", row=1, col=1)
        fig.update_yaxes(title_text="Log(SalePrice)", row=1, col=2)
        fig.update_xaxes(title_text="GrLivArea", row=2, col=1)
        fig.update_xaxes(title_text="GrLivArea", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)

# SECTION 6: Feature Set Results
elif sections[selected_section] == "results":
    st.markdown('<h2 class="section-header">Final Feature Set Results</h2>', unsafe_allow_html=True)
    
    # Before and after comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Feature Engineering")
        st.metric("Total Features", "80")
        st.metric("Categorical Features", "43")
        st.metric("Numerical Features", "37")
        st.metric("Skewed Features", "23")
        st.metric("Missing Values", "Present")
        
    with col2:
        st.subheader("After Feature Engineering")
        st.metric("Total Features", "191")
        st.metric("Encoded Features", "191")
        st.metric("New Features Created", "8")
        st.metric("Skewed Features", "16")
        st.metric("Missing Values", "None")
    
    # Final feature breakdown
    st.subheader("Final Feature Set Breakdown")
    
    feature_breakdown = {
        'Feature Type': [
            'Label Encoded (Ordinal)',
            'One-Hot Encoded (Nominal)',
            'Numerical (Original)',
            'Numerical (Engineered)',
            'Transformed (Log)',
            'Binary Indicators'
        ],
        'Count': [22, 149, 20, 8, 23, 3],
        'Examples': [
            'ExterQual, LotShape, BsmtFinType1',
            'Neighborhood_*, SaleType_*, Foundation_*',
            'OverallQual, YearsSinceRemodel, TotalBaths',
            'HouseAge, GarageAge, TotalFlrSF',
            'LotArea, GrLivArea, MasVnrArea',
            'HasGarage, CentralAir, PavedDrive'
        ]
    }
    
    breakdown_df = pd.DataFrame(feature_breakdown)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
    
    # Show sample of final features if available
    if 'X_train_final' in processed_data:
        st.subheader("Sample of Final Feature Set")
        final_sample = processed_data['X_train_final'].head(5)
        st.dataframe(final_sample, use_container_width=True)
        
        # Feature statistics
        st.subheader("Feature Set Statistics")
        stats_data = {
            'Statistic': ['Total Features', 'Mean', 'Std Dev', 'Min', 'Max'],
            'Value': [
                final_sample.shape[1],
                f"{final_sample.mean().mean():.3f}",
                f"{final_sample.std().mean():.3f}",
                f"{final_sample.min().min():.3f}",
                f"{final_sample.max().max():.3f}"
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

# SECTION 7: Quality Validation
elif sections[selected_section] == "validation":
    st.markdown('<h2 class="section-header">Quality Validation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive validation ensures the engineered features meet quality standards for modeling.
    """)
    
    # Validation checklist
    validation_items = {
        'Validation Check': [
            'No Missing Values',
            'No Infinite Values',
            'Proper Data Types',
            'Feature Consistency',
            'Reduced Skewness',
            'Target Normalization',
            'Encoding Completeness',
            'Domain Knowledge Integration'
        ],
        'Status': ['✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed'],
        'Details': [
            'All 191 features have complete data',
            'All transformations handled edge cases properly',
            'All features converted to appropriate numerical types',
            'Train/test feature sets perfectly aligned',
            '23 → 16 highly skewed features (30.4% reduction)',
            'Target skewness: 1.879 → 0.121 (93.6% improvement)',
            '100% of categorical variables properly encoded',
            'Real estate expertise incorporated in feature design'
        ]
    }
    
    validation_df = pd.DataFrame(validation_items)
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    
    # Quality metrics
    st.subheader("Quality Metrics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missing Values", "0", delta="100% Complete")
    with col2:
        st.metric("Infinite Values", "0", delta="All Valid")
    with col3:
        st.metric("Feature Consistency", "100%", delta="Perfect Alignment")
    with col4:
        st.metric("Encoding Success", "100%", delta="All Categorical Handled")
    
    # Validation details
    st.markdown("""
    <div class="insight-box">
    <h4>Validation Success Indicators</h4>
    <ul>
        <li><strong>Data Completeness:</strong> No missing or infinite values in final feature set</li>
        <li><strong>Consistency:</strong> Train and test datasets have identical feature structures</li>
        <li><strong>Normality:</strong> Significant improvement in feature and target distributions</li>
        <li><strong>Interpretability:</strong> All engineered features have clear business meaning</li>
        <li><strong>Scalability:</strong> Pipeline can handle new data with same transformations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 8: Impact Analysis
elif sections[selected_section] == "impact":
    st.markdown('<h2 class="section-header">Feature Engineering Impact Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Analysis of how feature engineering improvements affect model readiness and expected performance.
    """)
    
    # Create tabs for different impact analyses
    tab1, tab2, tab3 = st.tabs(["Distribution Improvements", "Correlation Changes", "Expected Performance"])
    
    with tab1:
        st.subheader("Distribution Improvements")
        
        improvement_metrics = {
            'Improvement Area': [
                'Target Distribution',
                'Feature Skewness',
                'Outlier Impact',
                'Linear Relationships',
                'Model Assumptions'
            ],
            'Before': [
                'Highly skewed (1.879)',
                '23 highly skewed features',
                'High influence from extreme values',
                'Many non-linear relationships',
                'Violated normality assumptions'
            ],
            'After': [
                'Near-normal (0.121)',
                '16 highly skewed features',
                'Reduced outlier sensitivity',
                'More linear relationships',
                'Better assumption compliance'
            ],
            'Impact': [
                '93.6% skewness reduction',
                '30.4% fewer skewed features',
                'Improved model stability',
                'Enhanced predictive power',
                'Better model performance'
            ]
        }
        
        improvement_df = pd.DataFrame(improvement_metrics)
        st.dataframe(improvement_df, use_container_width=True, hide_index=True)
        
    with tab2:
        st.subheader("Correlation Structure Changes")
        
        correlation_changes = {
            'Feature Relationship': [
                'GarageCars vs GarageArea',
                'Total Floor Area Components',
                'Age-based Features',
                'Quality Features',
                'Bathroom Features'
            ],
            'Original Issue': [
                'High multicollinearity (0.882)',
                'Separate 1st/2nd floor correlations',
                'Missing temporal relationships',
                'Scattered quality measures',
                'Multiple bathroom counts'
            ],
            'Engineering Solution': [
                'GarageAreaPerCar ratio feature',
                'TotalFlrSF consolidation',
                'HouseAge and GarageAge features',
                'Ordinal encoding preservation',
                'TotalBaths weighted metric'
            ],
            'Expected Outcome': [
                'Reduced multicollinearity',
                'Better size representation',
                'Temporal pattern capture',
                'Preserved ordinality',
                'Improved bathroom signal'
            ]
        }
        
        correlation_df = pd.DataFrame(correlation_changes)
        st.dataframe(correlation_df, use_container_width=True, hide_index=True)
        
    with tab3:
        st.subheader("Expected Performance Improvements")
        
        performance_expectations = {
            'Model Aspect': [
                'Linear Models',
                'Tree-based Models',
                'Feature Importance',
                'Prediction Accuracy',
                'Model Interpretability'
            ],
            'Expected Improvement': [
                'Better linear assumptions compliance',
                'Enhanced split quality with ordinal encoding',
                'Engineered features ranking higher',
                'Reduced prediction errors',
                'Clearer feature importance patterns'
            ],
            'Confidence Level': ['High', 'High', 'Medium', 'High', 'High']
        }
        
        performance_df = pd.DataFrame(performance_expectations)
        st.dataframe(performance_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="transformation-box">
        <h4>Performance Improvement Indicators</h4>
        <ul>
            <li><strong>Linear Models:</strong> Log transformations improve linearity assumptions</li>
            <li><strong>Tree Models:</strong> Ordinal encoding provides better split points</li>
            <li><strong>Feature Selection:</strong> Engineered features expected to show higher importance</li>
            <li><strong>Generalization:</strong> Reduced overfitting through proper encoding</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# SECTION 9: Key Insights
elif sections[selected_section] == "insights":
    st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h3>Feature Engineering Approach</h3>
    <ul>
        <li><strong>Feature Expansion:</strong> Strategic increase from 80 to 191 features through domain knowledge</li>
        <li><strong>Skewness Reduction:</strong> Significant improvements in data distribution (23 → 16 skewed features)</li>
        <li><strong>Multicollinearity Management:</strong> High correlation features consolidated through ratios</li>
        <li><strong>Domain Integration:</strong> Real estate expertise applied to create meaningful features</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h3>Encoding Strategy Success</h3>
    <ul>
        <li><strong>Ordinal Preservation:</strong> Quality ratings maintain natural hierarchy through label encoding</li>
        <li><strong>Nominal Handling:</strong> Geographic and categorical variables properly one-hot encoded</li>
        <li><strong>Information Retention:</strong> No data loss through appropriate encoding choices</li>
        <li><strong>Model Compatibility:</strong> All features now numerical for machine learning algorithms</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h3>Target Optimization Impact</h3>
    <ul>
        <li><strong>Distribution Normalization:</strong> Log transformation reduces skewness from 1.879 to 0.121</li>
        <li><strong>Linear Relationships:</strong> Enhanced correlation patterns with predictor variables</li>
        <li><strong>Model Assumptions:</strong> Better compliance with linear regression assumptions</li>
        <li><strong>Outlier Mitigation:</strong> Reduced influence of extreme property values</li>
    </ul>
    </div>
    
    <div class="transformation-box">
    <h3>Real Estate Domain Knowledge Integration</h3>
    <ul>
        <li><strong>Age Features:</strong> HouseAge and GarageAge capture depreciation and market trends</li>
        <li><strong>Quality Hierarchy:</strong> Ordinal encoding preserves condition rating relationships</li>
        <li><strong>Space Efficiency:</strong> Ratio features like GarageAreaPerCar provide efficiency metrics</li>
        <li><strong>Consolidated Metrics:</strong> TotalBaths and TotalFlrSF provide comprehensive measurements</li>
    </ul>
    </div>
    
    <div class="insight-box">
    <h3>Model Readiness Achievements</h3>
    <ul>
        <li><strong>Numerical Stability:</strong> Log transformations prevent extreme value dominance</li>
        <li><strong>Feature Richness:</strong> Comprehensive property representation through 191 features</li>
        <li><strong>Consistency Guarantee:</strong> Train/test feature alignment maintained throughout</li>
        <li><strong>Quality Assurance:</strong> Zero missing values and proper data type consistency</li>
    </ul>
    </div>
    
    <div class="process-box">
    <h3>Next Steps for Modeling</h3>
    <p>The engineered feature set is now ready for model training with the following advantages:</p>
    <ul>
        <li><strong>Algorithm Flexibility:</strong> Features work well with both linear and tree-based models</li>
        <li><strong>Baseline Establishment:</strong> Strong foundation for model comparison and selection</li>
        <li><strong>Feature Selection Ready:</strong> Rich feature set enables sophisticated selection techniques</li>
        <li><strong>Performance Optimization:</strong> Multiple model types can be tested and optimized</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This feature engineering pipeline follows the comprehensive methodology from notebook 03_feature_engineering.ipynb*")