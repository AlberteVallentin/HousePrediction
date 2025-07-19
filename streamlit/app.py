import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the Python path to import utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="House Price Prediction Analysis - Ames Housing Dataset",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 2rem 0;
        background: linear-gradient(90deg, #f0f4ff 0%, #e8f2ff 100%);
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
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
    
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        color: #666;
        line-height: 1.6;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .nav-button {
        display: block;
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #ffffff;
        border: 2px solid #3498db;
        border-radius: 0.5rem;
        color: #3498db;
        text-decoration: none;
        text-align: center;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background-color: #3498db;
        color: white;
    }
    
    .highlight-box {
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
    
    .footer {
        background-color: #2c3e50;
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-top: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">House Price Prediction Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comprehensive analysis of the Ames Housing dataset using machine learning techniques</p>', unsafe_allow_html=True)

# Load basic dataset info for dashboard
@st.cache_data
def load_dataset_overview():
    """Load basic dataset information for the overview."""
    try:
        # Try to load from uploaded files first
        train_path = None
        test_path = None
        
        # Check for uploaded files in session state
        if hasattr(st.session_state, 'uploaded_files'):
            for file_name, file_content in st.session_state.uploaded_files.items():
                if 'train' in file_name.lower() and file_name.endswith('.csv'):
                    train_df = pd.read_csv(file_content)
                elif 'test' in file_name.lower() and file_name.endswith('.csv'):
                    test_df = pd.read_csv(file_content)
        
        # Fallback to local files
        if 'train_df' not in locals():
            base_path = os.path.join(os.path.dirname(__file__), '..')
            train_path = os.path.join(base_path, 'data', 'raw', 'train.csv')
            test_path = os.path.join(base_path, 'data', 'raw', 'test.csv')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
            else:
                return None
        
        return {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': train_df.shape[1],
            'price_range': (train_df['SalePrice'].min(), train_df['SalePrice'].max()),
            'avg_price': train_df['SalePrice'].mean(),
            'missing_values': train_df.isnull().sum().sum()
        }
    except Exception as e:
        return None

# Sidebar navigation
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("### Analysis Workflow")

pages = [
    ("Data Exploration", "Explore the dataset structure, distributions, and patterns"),
    ("Data Preprocessing", "Learn about data cleaning and preparation steps"),
    ("Feature Engineering", "Discover feature creation and selection processes"),
    ("Model Development", "Compare different models and their performance"),
    ("Price Predictor", "Interactive tool for predicting house prices"),
    ("AI Assistant", "Chat with AI about the project and findings")
]

selected_page = st.sidebar.radio(
    "Choose a section:",
    [page[0] for page in pages],
    format_func=lambda x: x
)

# Show description of selected page
page_descriptions = {page[0]: page[1] for page in pages}
st.sidebar.markdown(f"**{selected_page}:** {page_descriptions[selected_page]}")

# Load dataset overview
dataset_info = load_dataset_overview()

# Main content area
if selected_page == "Data Exploration":
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This application presents a comprehensive analysis of house prices using the Ames Housing dataset. 
    The project demonstrates a complete machine learning pipeline from data exploration to model deployment, 
    following industry best practices for data science workflows.
    """)
    
    # Project goals
    st.markdown('<h3 class="section-header">Project Objectives</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Data Analysis Goals</div>
            <div class="card-description">
                <ul>
                    <li>Analyze housing market patterns and price drivers</li>
                    <li>Identify key features that influence property values</li>
                    <li>Understand data quality issues and preprocessing needs</li>
                    <li>Explore relationships between features and target variable</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Technical Objectives</div>
            <div class="card-description">
                <ul>
                    <li>Build robust predictive models for house prices</li>
                    <li>Implement feature engineering and selection techniques</li>
                    <li>Compare multiple machine learning algorithms</li>
                    <li>Create an interactive deployment-ready application</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset overview with metrics
    st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)
    
    if dataset_info:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{dataset_info['train_samples']:,}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{dataset_info['test_samples']:,}</div>
                <div class="metric-label">Test Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{dataset_info['features']}</div>
                <div class="metric-label">Features</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_price = dataset_info['avg_price']
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${avg_price:,.0f}</div>
                <div class="metric-label">Average Price</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional dataset insights
        st.markdown("""
        <div class="highlight-box">
        <strong>Dataset Highlights:</strong><br>
        The Ames Housing dataset contains detailed information about residential properties in Ames, Iowa, 
        covering various aspects including location, size, quality, and amenities. This comprehensive dataset 
        provides an excellent foundation for developing robust price prediction models.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="warning-box">
        <strong>Dataset Loading:</strong><br>
        Dataset information will be displayed once the data files are available. 
        The application supports both uploaded files and local data storage.
        </div>
        """, unsafe_allow_html=True)
    
    # Methodology overview
    st.markdown('<h3 class="section-header">Analysis Methodology</h3>', unsafe_allow_html=True)
    
    methodology_steps = [
        ("Data Exploration", "Comprehensive analysis of data structure, distributions, missing values, and feature relationships"),
        ("Data Preprocessing", "Cleaning, missing value treatment, outlier detection, and data type corrections"),
        ("Feature Engineering", "Creating new features, transformations, encoding, and feature selection"),
        ("Model Development", "Training multiple algorithms, hyperparameter tuning, and performance evaluation"),
        ("Model Deployment", "Creating an interactive application for real-time predictions")
    ]
    
    for i, (step, description) in enumerate(methodology_steps, 1):
        st.markdown(f"""
        <div class="card">
            <div class="card-title">{i}. {step}</div>
            <div class="card-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    # For other pages, show navigation message
    st.markdown('<h2 class="section-header">Navigation</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="highlight-box">
    <strong>Currently viewing: {selected_page}</strong><br>
    Use the sidebar navigation to switch between different sections of the analysis.
    Each section provides detailed insights into different aspects of the house price prediction project.
    </div>
    """, unsafe_allow_html=True)
    
    # Show workflow visualization
    st.markdown('<h3 class="section-header">Analysis Workflow</h3>', unsafe_allow_html=True)
    
    workflow_descriptions = {
        "Data Exploration": """
        **Explore the Dataset**
        - Understand data structure and dimensions
        - Analyze target variable distribution
        - Identify missing values and data quality issues
        - Examine feature correlations and relationships
        - Discover neighborhood and categorical patterns
        """,
        
        "Data Preprocessing": """
        **Clean and Prepare Data**
        - Handle missing values systematically
        - Detect and treat outliers
        - Correct data types and inconsistencies
        - Prepare data for feature engineering
        - Document preprocessing decisions
        """,
        
        "Feature Engineering": """
        **Enhance and Select Features**
        - Create new informative features
        - Apply transformations for better modeling
        - Encode categorical variables appropriately
        - Select most relevant features
        - Reduce multicollinearity issues
        """,
        
        "Model Development": """
        **Build and Evaluate Models**
        - Train multiple machine learning algorithms
        - Perform hyperparameter optimization
        - Evaluate model performance using cross-validation
        - Compare different modeling approaches
        - Select best performing model
        """,
        
        "Price Predictor": """
        **Interactive Prediction Tool**
        - Input house characteristics
        - Get real-time price predictions
        - Understand feature importance
        - Explore different scenarios
        - Validate model decisions
        """,
        
        "AI Assistant": """
        **Intelligent Project Assistant**
        - Ask questions about the analysis
        - Get insights into methodology
        - Understand model results
        - Explore feature relationships
        - Learn about data science concepts
        """
    }
    
    if selected_page in workflow_descriptions:
        st.markdown(workflow_descriptions[selected_page])
    
    # Navigation buttons
    st.markdown('<h3 class="section-header">Quick Access</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start with Data Exploration", use_container_width=True):
            st.sidebar.selectbox("Choose a section:", [page[0] for page in pages], index=0)
    
    with col2:
        if st.button("Jump to Price Predictor", use_container_width=True):
            st.sidebar.selectbox("Choose a section:", [page[0] for page in pages], index=4)
    
    with col3:
        if st.button("Chat with AI Assistant", use_container_width=True):
            st.sidebar.selectbox("Choose a section:", [page[0] for page in pages], index=5)

# Key findings summary (always visible)
st.markdown('<h2 class="section-header">Project Summary</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">Data Quality Assessment</div>
        <div class="card-description">
            <ul>
                <li><strong>1,460</strong> training samples with <strong>81</strong> features</li>
                <li><strong>6.6%</strong> missing values, mostly in optional features</li>
                <li><strong>34</strong> data quality issues identified across <strong>31</strong> properties</li>
                <li><strong>2</strong> problematic outliers requiring special treatment</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">Modeling Results</div>
        <div class="card-description">
            <ul>
                <li><strong>12</strong> different algorithms tested and compared</li>
                <li><strong>Ensemble methods</strong> achieved best performance</li>
                <li><strong>Log transformation</strong> improved target variable normality</li>
                <li><strong>Feature engineering</strong> significantly enhanced predictions</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Technical implementation details
st.markdown('<h3 class="section-header">Technical Implementation</h3>', unsafe_allow_html=True)

implementation_details = """
**Technologies Used:**
- **Python 3.8+** for data analysis and modeling
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for machine learning algorithms
- **Plotly** for interactive visualizations
- **Streamlit** for web application development

**Key Features:**
- Interactive data exploration with dynamic visualizations
- Comprehensive preprocessing pipeline with detailed logging
- Advanced feature engineering with automated selection
- Multiple model comparison with cross-validation
- Real-time prediction interface with explanation capabilities
"""

st.markdown(implementation_details)

# Footer
st.markdown("""
<div class="footer">
    <h3>House Price Prediction Analysis</h3>
    <p><strong>Data Science Project</strong> | <strong>Ames Housing Dataset</strong></p>
    <p>This application demonstrates a complete machine learning pipeline including data exploration, 
    preprocessing, feature engineering, model development, and deployment.</p>
    <p><em>Built with Streamlit, Scikit-learn, and modern data science best practices</em></p>
</div>
""", unsafe_allow_html=True)