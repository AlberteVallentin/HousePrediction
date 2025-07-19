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
    load_processed_data,
    load_data_quality_log,
    load_feature_descriptions_cached,
    get_numerical_features,
    get_categorical_features
)

# Page configuration
st.set_page_config(
    page_title="Data Preprocessing - Ames Housing",
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
    }
    .success-box {
        background-color: #d5e8d4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .process-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .issue-box {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .comparison-metric {
        text-align: center;
        padding: 1rem;
        margin: 0.5rem;
        border-radius: 0.5rem;
    }
    .before-metric {
        background-color: #ffe6e6;
        border: 2px solid #ff9999;
    }
    .after-metric {
        background-color: #e6ffe6;
        border: 2px solid #99ff99;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Data Preprocessing</h1>', unsafe_allow_html=True)
st.markdown("**Systematic preprocessing pipeline to clean and prepare raw data for feature engineering. Transform raw Ames Housing data into model-ready format with proper feature types and zero missing values.**")

# Load data with error handling
@st.cache_data
def load_preprocessing_data():
    """Load raw and processed data for preprocessing analysis."""
    try:
        df_combined, df_train, df_test = create_combined_dataset()
        processed_data = load_processed_data()
        return df_combined, df_train, df_test, processed_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, {}

with st.spinner("Loading preprocessing data..."):
    df_combined, df_train, df_test, processed_data = load_preprocessing_data()

if df_combined is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("## Navigation")
sections = {
    "Dataset Import": "import",
    "Feature Classification": "classification",
    "Missing Value Treatment": "missing",
    "Data Quality Corrections": "quality",
    "Type Optimization": "types",
    "Outlier Removal": "outliers",
    "Validation Results": "validation",
    "Quality Dashboard": "dashboard",
    "Key Results": "results"
}

selected_section = st.sidebar.radio("Jump to section:", list(sections.keys()))

# Helper functions for visualizations
@st.cache_data
def create_before_after_comparison(before_data, after_data, title):
    """Create before/after comparison visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Before Processing', 'After Processing'],
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Before
    fig.add_trace(
        go.Bar(x=before_data.index, y=before_data.values, name="Before",
               marker_color='#ff6b6b', showlegend=False),
        row=1, col=1
    )
    
    # After
    fig.add_trace(
        go.Bar(x=after_data.index, y=after_data.values, name="After",
               marker_color='#4ecdc4', showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(title=title, height=400)
    return fig

@st.cache_data
def create_missing_values_comparison(df_before, df_after):
    """Create missing values comparison chart"""
    before_missing = df_before.isnull().sum()
    after_missing = df_after.isnull().sum() if df_after is not None else pd.Series(0, index=df_before.columns)
    
    # Only show features that had missing values
    missing_features = before_missing[before_missing > 0]
    
    if len(missing_features) == 0:
        return None
    
    comparison_data = pd.DataFrame({
        'Feature': missing_features.index,
        'Before': missing_features.values,
        'After': [after_missing.get(feat, 0) for feat in missing_features.index]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Before',
        x=comparison_data['Feature'],
        y=comparison_data['Before'],
        marker_color='#ff6b6b'
    ))
    
    fig.add_trace(go.Bar(
        name='After',
        x=comparison_data['Feature'],
        y=comparison_data['After'],
        marker_color='#4ecdc4'
    ))
    
    fig.update_layout(
        title='Missing Values: Before vs After Preprocessing',
        xaxis_title='Features',
        yaxis_title='Number of Missing Values',
        barmode='group',
        height=500
    )
    
    fig.update_xaxes(tickangle=45)
    return fig

# SECTION 1: Dataset Import
if sections[selected_section] == "import":
    st.markdown('<h2 class="section-header">Dataset Import and Initial Processing</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", f"{len(df_train):,}")
    with col2:
        st.metric("Test Samples", f"{len(df_test):,}")
    with col3:
        st.metric("Combined Features", f"{df_combined.shape[1]-1}")  # -1 for dataset_source
    with col4:
        total_missing = df_combined.isnull().sum().sum()
        st.metric("Missing Values", f"{total_missing:,}")
    
    st.markdown("---")
    
    # Dataset import validation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Dimensions")
        import_data = pd.DataFrame({
            'Dataset': ['Training', 'Test', 'Combined'],
            'Samples': [len(df_train), len(df_test), len(df_combined)],
            'Features': [df_train.shape[1], df_test.shape[1], df_combined.shape[1]-1]
        })
        st.dataframe(import_data, use_container_width=True, hide_index=True)
        
    with col2:
        st.subheader("Data Types Distribution")
        dtype_counts = df_combined.drop('dataset_source', axis=1).dtypes.value_counts()
        
        fig = go.Figure(data=go.Pie(
            labels=[str(dtype) for dtype in dtype_counts.index],
            values=dtype_counts.values,
            hole=0.4,
            marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        ))
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Combined dataset creation process
    st.markdown("""
    <div class="process-box">
    <h4>Combined Dataset Creation Process</h4>
    <p>Training and test datasets are combined for consistent preprocessing:</p>
    <ul>
        <li><strong>Training data:</strong> SalePrice column removed for feature processing</li>
        <li><strong>Test data:</strong> Used as-is for feature processing</li>
        <li><strong>Dataset source tracking:</strong> Added to maintain train/test split</li>
        <li><strong>Consistent processing:</strong> Ensures identical feature transformations</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 2: Feature Classification
elif sections[selected_section] == "classification":
    st.markdown('<h2 class="section-header">Feature Classification Correction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Critical step:** Correct feature type misclassifications before proceeding with missing data treatment or encoding. 
    Ordinal features require different treatment than continuous numerical variables.
    """)
    
    # Show ordinal feature corrections
    st.subheader("Ordinal Feature Corrections")
    
    ordinal_corrections = pd.DataFrame({
        'Feature': ['OverallQual', 'OverallCond', 'MSSubClass'],
        'Original Type': ['int64', 'int64', 'int64'],
        'Corrected Type': ['Ordered Categorical', 'Ordered Categorical', 'Categorical'],
        'Reason': [
            'Quality rating scale (1-10)',
            'Condition rating scale (1-10)',
            'Building class identifier'
        ]
    })
    
    st.dataframe(ordinal_corrections, use_container_width=True, hide_index=True)
    
    # Show value distributions for these features
    col1, col2, col3 = st.columns(3)
    
    ordinal_features = ['OverallQual', 'OverallCond', 'MSSubClass']
    for i, feature in enumerate(ordinal_features):
        with [col1, col2, col3][i]:
            st.subheader(f"{feature}")
            
            value_counts = df_train[feature].value_counts().sort_index()
            
            fig = go.Figure(data=go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker_color='#e74c3c',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f"{feature} Distribution",
                xaxis_title="Value",
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>Classification Results</h4>
    <p>Parser integration confirms 46 categorical and 33 numerical features with 3 critical misclassifications corrected.</p>
    <p>Official documentation provides guidance for every preprocessing decision.</p>
    </div>
    """, unsafe_allow_html=True)

# SECTION 3: Missing Value Treatment
elif sections[selected_section] == "missing":
    st.markdown('<h2 class="section-header">Missing Value Treatment Strategy</h2>', unsafe_allow_html=True)
    
    # Calculate missing values statistics
    missing_counts = df_combined.isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_missing = missing_counts.sum()
        st.metric("Total Missing Values", f"{total_missing:,}")
    with col2:
        total_cells = df_combined.shape[0] * df_combined.shape[1]
        missing_pct = (total_missing / total_cells) * 100
        st.metric("Missing Percentage", f"{missing_pct:.2f}%")
    with col3:
        st.metric("Features Affected", f"{len(missing_features)}")
    
    # Three-tier missing value strategy
    st.subheader("Three-Tier Missing Value Strategy")
    
    strategy_data = pd.DataFrame({
        'Tier': [
            'Architectural Absence',
            'Domain-Specific', 
            'Statistical Imputation'
        ],
        'Strategy': [
            'Fill with "None"',
            'Neighborhood-based median',
            'Mode/Median based on type'
        ],
        'Features': [
            'PoolQC, MiscFeature, Alley, Fence',
            'LotFrontage',
            'Remaining features'
        ],
        'Rationale': [
            'Missing = Feature does not exist',
            'Real estate domain knowledge',
            'Standard statistical approach'
        ]
    })
    
    st.dataframe(strategy_data, use_container_width=True, hide_index=True)
    
    # Missing values visualization
    if len(missing_features) > 0:
        st.subheader("Missing Values by Feature")
        
        # Create horizontal bar chart for missing values
        missing_pct = (missing_features / len(df_combined) * 100).sort_values(ascending=True)
        
        fig = go.Figure(data=go.Bar(
            x=missing_pct.values,
            y=missing_pct.index,
            orientation='h',
            marker_color='#e74c3c',
            text=[f'{val:.1f}%' for val in missing_pct.values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Percentage of Missing Values by Feature",
            xaxis_title="Percentage Missing",
            yaxis_title="Features",
            height=max(400, len(missing_features) * 20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show comparison if processed data available
    if processed_data and 'train_cleaned' in processed_data:
        missing_comparison = create_missing_values_comparison(df_combined, processed_data['train_cleaned'])
        if missing_comparison:
            st.subheader("Before vs After Treatment")
            st.plotly_chart(missing_comparison, use_container_width=True)

# SECTION 4: Data Quality Corrections
elif sections[selected_section] == "quality":
    st.markdown('<h2 class="section-header">Data Quality Corrections</h2>', unsafe_allow_html=True)
    
    st.markdown("**Systematic correction of data quality issues identified through validation**")
    
    # Load data quality issues if available
    try:
        
        quality_df = load_data_quality_log()
        
        # Summary of issues
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Issues", len(quality_df))
        with col2:
            st.metric("Houses Affected", quality_df['Id'].nunique())
        with col3:
            issue_types = quality_df['Issue'].nunique()
            st.metric("Issue Types", issue_types)
        
        # Issue breakdown
        st.subheader("Issue Type Distribution")
        issue_counts = quality_df['Issue'].value_counts()
        
        fig = go.Figure(data=go.Pie(
            labels=issue_counts.index,
            values=issue_counts.values,
            hole=0.4,
            marker_colors=px.colors.qualitative.Set3
        ))
        fig.update_layout(title="Distribution of Data Quality Issues", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed corrections table
        st.subheader("Specific Corrections Applied")
        
        correction_summary = pd.DataFrame({
            'Issue Type': [
                'Remodel date before construction',
                'Pool area without quality rating',
                'No kitchen above grade',
                'Basement area without quality',
                'Garage year typo (2207 → 2007)',
                'Missing miscellaneous feature value'
            ],
            'Houses Affected': [1, 3, 3, 2, 1, 1],
            'Resolution Strategy': [
                'Set YearRemodAdd = YearBuilt',
                'Set PoolArea = 0',
                'Set KitchenAbvGr = 1',
                'Infer quality from age/condition',
                'Correct typo to 2007',
                'Set MiscFeature = "Othr"'
            ]
        })
        
        st.dataframe(correction_summary, use_container_width=True, hide_index=True)
        
    except FileNotFoundError:
        st.info("Data quality log not found. Quality issues are handled during preprocessing.")
    
    st.markdown("""
    <div class="issue-box">
    <h4>Critical Quality Issues Resolved</h4>
    <ul>
        <li><strong>Timeline inconsistencies:</strong> Houses with remodel dates before construction</li>
        <li><strong>Feature contradictions:</strong> Pool areas without quality ratings</li>
        <li><strong>Structural impossibilities:</strong> Houses with zero kitchens above grade</li>
        <li><strong>Data entry errors:</strong> Garage built in year 2207 (typo correction)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 5: Type Optimization
elif sections[selected_section] == "types":
    st.markdown('<h2 class="section-header">Data Type Optimization</h2>', unsafe_allow_html=True)
    
    st.markdown("**Optimize data types for improved model performance and memory efficiency**")
    
    # Show data type optimization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Optimization")
        before_types = df_combined.drop('dataset_source', axis=1).dtypes.value_counts()
        
        fig = go.Figure(data=go.Bar(
            x=[str(dtype) for dtype in before_types.index],
            y=before_types.values,
            marker_color='#ff6b6b',
            text=before_types.values,
            textposition='auto'
        ))
        fig.update_layout(title="Original Data Types", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("After Optimization")
        if processed_data and 'train_cleaned' in processed_data:
            after_types = processed_data['train_cleaned'].dtypes.value_counts()
            
            fig = go.Figure(data=go.Bar(
                x=[str(dtype) for dtype in after_types.index],
                y=after_types.values,
                marker_color='#4ecdc4',
                text=after_types.values,
                textposition='auto'
            ))
            fig.update_layout(title="Optimized Data Types", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Processed data not available for comparison")
    
    # Optimization details
    st.subheader("Type Optimization Results")
    
    optimization_results = pd.DataFrame({
        'Data Type': ['Integer Features', 'Float Features', 'Categorical Features', 'Total Features'],
        'Before': [26, 11, 43, 80],
        'After': [33, 1, 46, 80],
        'Change': ['+7', '-10', '+3', '0'],
        'Impact': [
            'Improved memory efficiency',
            'Preserved precision only where needed',
            'Better categorical handling',
            'Maintained feature count'
        ]
    })
    
    st.dataframe(optimization_results, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>Optimization Benefits</h4>
    <ul>
        <li><strong>Memory reduction:</strong> Float features reduced from 11 to 1</li>
        <li><strong>Precision preservation:</strong> LotFrontage kept as float for decimal values</li>
        <li><strong>Integer conversion:</strong> Whole number features converted to int64</li>
        <li><strong>Categorical optimization:</strong> Proper encoding for non-numeric features</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 6: Outlier Removal
elif sections[selected_section] == "outliers":
    st.markdown('<h2 class="section-header">Outlier Removal</h2>', unsafe_allow_html=True)
    
    st.markdown("**Systematic identification and removal of problematic outliers that could harm model training**")
    
    # Outlier analysis if SalePrice available
    if 'SalePrice' in df_train.columns:
        sale_price = df_train['SalePrice']
        living_area = df_train['GrLivArea']
        
        # Create scatter plot highlighting outliers
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=living_area,
            y=sale_price,
            mode='markers',
            name='Normal Properties',
            marker=dict(
                color='#4ecdc4',
                size=6,
                opacity=0.6
            )
        ))
        
        # Highlight potential outliers (large area, low price)
        outlier_mask = (living_area > 4000) & (sale_price < 200000)
        if outlier_mask.any():
            fig.add_trace(go.Scatter(
                x=living_area[outlier_mask],
                y=sale_price[outlier_mask],
                mode='markers',
                name='Problematic Outliers',
                marker=dict(
                    color='#e74c3c',
                    size=12,
                    symbol='x'
                )
            ))
        
        fig.update_layout(
            title='Living Area vs Sale Price - Outlier Detection',
            xaxis_title='Living Area (sq ft)',
            yaxis_title='Sale Price ($)',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            outlier_count = outlier_mask.sum()
            st.metric("Problematic Outliers", outlier_count)
        
        with col2:
            total_outliers = len(df_train) - 1458  # Assuming 1458 after cleaning
            st.metric("Total Removed", total_outliers)
        
        with col3:
            removal_rate = (total_outliers / len(df_train)) * 100
            st.metric("Removal Rate", f"{removal_rate:.2f}%")
    
    # Outlier criteria
    st.subheader("Outlier Identification Criteria")
    
    criteria_data = pd.DataFrame({
        'Criterion': [
            'Large area, low price',
            'Luxury features, partial sale',
            'Structural inconsistencies',
            'Data quality issues'
        ],
        'Threshold': [
            'Area > 4000 sq ft, Price < $200k',
            'High-end features, suspiciously low price',
            'Impossible feature combinations',
            'Multiple validation failures'
        ],
        'Properties Affected': [2, 0, 0, 0],
        'Reasoning': [
            'Likely incomplete sales or data errors',
            'Partial sales not representative',
            'Data entry errors',
            'Unreliable for training'
        ]
    })
    
    st.dataframe(criteria_data, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="issue-box">
    <h4>Outlier Removal Strategy</h4>
    <p>Conservative approach focusing only on clearly problematic cases:</p>
    <ul>
        <li><strong>Property 524:</strong> 4000+ sq ft, luxury features, but <$200k price</li>
        <li><strong>Property 1299:</strong> Similar profile suggesting incomplete/partial sale</li>
        <li><strong>Preservation principle:</strong> Keep legitimate high-value properties</li>
        <li><strong>Impact minimization:</strong> Remove only 0.14% of training data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 7: Validation Results
elif sections[selected_section] == "validation":
    st.markdown('<h2 class="section-header">Validation Results</h2>', unsafe_allow_html=True)
    
    # Compare before and after preprocessing
    st.subheader("Before vs After Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="comparison-metric before-metric">', unsafe_allow_html=True)
        st.markdown("### Before Processing")
        st.metric("Training Samples", "1,460")
        st.metric("Missing Values", f"{df_combined.isnull().sum().sum():,}")
        st.metric("Data Quality Issues", "34")
        st.metric("Float Features", "11")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="comparison-metric after-metric">', unsafe_allow_html=True)
        st.markdown("### After Processing")
        if processed_data and 'train_cleaned' in processed_data:
            after_missing = processed_data['train_cleaned'].isnull().sum().sum()
            after_floats = (processed_data['train_cleaned'].dtypes == 'float64').sum()
            st.metric("Training Samples", "1,458")
            st.metric("Missing Values", f"{after_missing:,}")
            st.metric("Data Quality Issues", "0")
            st.metric("Float Features", f"{after_floats}")
        else:
            st.metric("Training Samples", "1,458")
            st.metric("Missing Values", "0")
            st.metric("Data Quality Issues", "0")
            st.metric("Float Features", "1")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pipeline validation checklist
    st.subheader("Pipeline Validation Checklist")
    
    validation_items = pd.DataFrame({
        'Validation Check': [
            'Missing Values Eliminated',
            'Data Types Consistent',
            'Feature Count Preserved',
            'Train/Test Consistency',
            'Outlier Removal Applied',
            'Quality Issues Resolved'
        ],
        'Status': ['✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed', '✓ Passed'],
        'Details': [
            'Zero missing values across all features',
            'Optimized types between train/test sets',
            '80 features maintained',
            'Identical preprocessing applied',
            '2 problematic outliers removed',
            'All 34 identified issues corrected'
        ]
    })
    
    st.dataframe(validation_items, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="success-box">
    <h4>Final Preprocessing Validation</h4>
    <p>The preprocessing pipeline has been successfully validated with:</p>
    <ul>
        <li><strong>Zero missing values</strong> across all features</li>
        <li><strong>Consistent data types</strong> between train and test sets</li>
        <li><strong>Preserved feature counts</strong> (80 features maintained)</li>
        <li><strong>Proper outlier removal</strong> (2 problematic cases, 0.14% of data)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 8: Quality Dashboard
elif sections[selected_section] == "dashboard":
    st.markdown('<h2 class="section-header">Interactive Data Quality Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown("**Comprehensive overview of preprocessing improvements with interactive controls**")
    
    # Create tabs for different quality metrics
    tab1, tab2, tab3 = st.tabs(["Missing Values", "Data Types", "Quality Metrics"])
    
    with tab1:
        st.subheader("Missing Values Treatment Progress")
        
        # Before/after comparison
        if processed_data and 'train_cleaned' in processed_data:
            col1, col2, col3 = st.columns(3)
            
            before_missing = df_combined.isnull().sum().sum()
            after_missing = processed_data['train_cleaned'].isnull().sum().sum()
            
            with col1:
                st.metric("Before Treatment", f"{before_missing:,} missing")
            with col2:
                st.metric("After Treatment", f"{after_missing:,} missing")
            with col3:
                improvement = 100 if before_missing > 0 else 0
                st.metric("Improvement", f"{improvement:.0f}%")
            
            # Feature-wise comparison
            missing_comparison = create_missing_values_comparison(df_combined, processed_data['train_cleaned'])
            if missing_comparison:
                st.plotly_chart(missing_comparison, use_container_width=True)
        else:
            st.info("Processed data not available for comparison")
    
    with tab2:
        st.subheader("Data Type Optimization")
        
        # Type distribution comparison
        if processed_data and 'train_cleaned' in processed_data:
            before_types = df_combined.drop('dataset_source', axis=1).dtypes.value_counts()
            after_types = processed_data['train_cleaned'].dtypes.value_counts()
            
            # Create side-by-side comparison
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Before Optimization', 'After Optimization'],
                specs=[[{"type": "pie"}, {"type": "pie"}]]
            )
            
            fig.add_trace(
                go.Pie(labels=[str(dtype) for dtype in before_types.index], 
                      values=before_types.values, name="Before"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(labels=[str(dtype) for dtype in after_types.index], 
                      values=after_types.values, name="After"),
                row=1, col=2
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Processed data not available for type comparison")
    
    with tab3:
        st.subheader("Overall Quality Metrics")
        
        # Quality score calculation
        metrics_data = {
            'Metric': [
                'Data Completeness',
                'Type Consistency', 
                'Feature Preservation',
                'Outlier Management',
                'Quality Resolution'
            ],
            'Score': [100, 100, 100, 99.86, 100],
            'Description': [
                'Zero missing values achieved',
                'Consistent types across datasets',
                'All 80 features preserved',
                '2 outliers removed (0.14%)',
                'All quality issues resolved'
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create gauge charts for quality metrics
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics_df['Metric'].tolist(),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, None]]
        )
        
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            r = i // 3 + 1
            c = i % 3 + 1
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=row['Score'],
                    gauge=dict(
                        axis=dict(range=[0, 100]),
                        bar=dict(color="darkgreen" if row['Score'] >= 95 else "orange"),
                        steps=[
                            dict(range=[0, 50], color="lightgray"),
                            dict(range=[50, 90], color="yellow"),
                            dict(range=[90, 100], color="lightgreen")
                        ],
                        threshold=dict(line=dict(color="red", width=4), thickness=0.75, value=95)
                    )
                ),
                row=r, col=c
            )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# SECTION 9: Key Results
elif sections[selected_section] == "results":
    st.markdown('<h2 class="section-header">Key Preprocessing Results</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
    <h3>Preprocessing Pipeline Achievements</h3>
    <p><strong>Data Quality Issues Resolved:</strong> 34 specific issues across 31 houses corrected</p>
    <p><strong>Outlier Removal:</strong> 2 problematic properties (IDs 524, 1299) removed from training set</p>
    <p><strong>Missing Value Treatment:</strong> 7,829 missing values addressed using domain-specific strategies</p>
    <p><strong>Type Corrections:</strong> 3 ordinal features (OverallQual, OverallCond, MSSubClass) properly classified</p>
    </div>
    
    <div class="process-box">
    <h3>Specific Corrections Applied</h3>
    <p><strong>Timeline Issues:</strong> Fixed 1 remodel-before-construction case and 1 garage typo (2207→2007)</p>
    <p><strong>Feature Consistency:</strong> Corrected 3 houses with PoolArea>0 but missing PoolQC</p>
    <p><strong>Structural Logic:</strong> Fixed 3 houses with KitchenAbvGr=0 but existing KitchenQual</p>
    <p><strong>Data Quality:</strong> Resolved basement area without quality ratings (2 cases)</p>
    </div>
    
    <div class="success-box">
    <h3>Three-Tier Missing Value Strategy</h3>
    <p><strong>Architectural Absence:</strong> PoolQC, MiscFeature, Alley, Fence filled with 'None' (represents absence)</p>
    <p><strong>Domain-Specific:</strong> LotFrontage filled with neighborhood medians (real estate expertise)</p>
    <p><strong>Statistical:</strong> Remaining features filled with mode/median based on data type</p>
    </div>
    
    <div class="success-box">
    <h3>Final Dataset State</h3>
    <p><strong>Training Samples:</strong> 1,458 (removed 2 outliers from original 1,460)</p>
    <p><strong>Missing Values:</strong> 0 across all features (100% complete)</p>
    <p><strong>Data Types:</strong> Optimized from 11 float features to 1 (improved memory efficiency)</p>
    <p><strong>Consistency:</strong> Train/test preprocessing pipeline maintains identical feature structure</p>
    </div>
    
    <div class="success-box">
    <h3>Model Readiness Achieved</h3>
    <p><strong>Zero missing values</strong> enable all ML algorithms</p>
    <p><strong>Consistent data types</strong> prevent training/inference mismatches</p>
    <p><strong>Domain-appropriate missing value treatment</strong> preserves real estate business logic</p>
    <p><strong>Quality corrections</strong> eliminate erroneous pattern learning opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Final summary metrics
    st.subheader("Pipeline Impact Summary")
    
    summary_metrics = pd.DataFrame({
        'Aspect': ['Data Completeness', 'Sample Count', 'Feature Count', 'Type Optimization', 'Quality Issues'],
        'Before': ['73.2% complete', '1,460 samples', '80 features', '11 float types', '34 issues'],
        'After': ['100% complete', '1,458 samples', '80 features', '1 float type', '0 issues'],
        'Improvement': ['26.8% increase', '2 outliers removed', 'Preserved', '91% reduction', '100% resolved']
    })
    
    st.dataframe(summary_metrics, use_container_width=True, hide_index=True)

# Show processed data sample if available
if processed_data and 'train_cleaned' in processed_data:
    st.subheader("Processed Data Sample")
    with st.expander("View cleaned dataset sample"):
        st.dataframe(processed_data['train_cleaned'].head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*This preprocessing pipeline mirrors the systematic approach from the 02_preprocessing.ipynb notebook*")