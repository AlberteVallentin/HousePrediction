import streamlit as st
import sys
import os

# Add the parent directory to the Python path to import notebooks utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from notebooks.data_description_parser import load_feature_descriptions

# Page configuration
st.set_page_config(
    page_title="House Price Prediction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("House Price Prediction Analysis")
st.markdown("---")

# Introduction section
st.header("Project Overview")
st.markdown("""
This application presents a comprehensive analysis of house prices using the Ames Housing dataset. 
The project demonstrates a complete machine learning pipeline from data exploration to model deployment.

**Project Goals:**
- Analyze housing market patterns and price drivers
- Build robust predictive models for house prices
- Create an interactive tool for price predictions
- Demonstrate best practices in data science workflow

**Dataset:** The Ames Housing dataset contains 1,460 training samples and 1,459 test samples with 81 features 
covering various aspects of residential properties in Ames, Iowa.
""")

# Dataset description
st.header("Dataset Description")
st.markdown("""
The dataset includes detailed information about residential properties including:
- **Physical characteristics:** Size, age, condition, and quality ratings
- **Location factors:** Neighborhood, lot configuration, and proximity conditions  
- **Structural features:** Foundation, heating, electrical systems, and building materials
- **Amenities:** Garages, basements, porches, pools, and other features
- **Sales information:** Sale price, date, type, and conditions
""")

# Load and display feature descriptions
try:
    feature_descriptions = load_feature_descriptions("../docs/data_description.txt")
    
    st.subheader("Feature Categories")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Features", len(feature_descriptions))
    
    with col2:
        categorical_count = len(feature_descriptions[feature_descriptions['Type'] == 'Categorical'])
        st.metric("Categorical Features", categorical_count)
    
    with col3:
        numerical_count = len(feature_descriptions[feature_descriptions['Type'] == 'Numerical'])
        st.metric("Numerical Features", numerical_count)
    
    # Feature overview table
    st.subheader("Feature Overview")
    st.markdown("The following table shows all features in the dataset with their types and descriptions:")
    
    # Create a more streamlined display
    display_df = feature_descriptions.copy()
    display_df['Description'] = display_df['Description'].apply(
        lambda x: x[:80] + "..." if len(x) > 80 else x
    )
    
    st.dataframe(
        display_df[['Feature', 'Type', 'Description']],
        use_container_width=True,
        height=400
    )
    
except Exception as e:
    st.error(f"Error loading feature descriptions: {str(e)}")

# Navigation section
st.header("Navigation")
st.markdown("""
Use the sidebar to navigate between different sections of the analysis:

1. **🔍 Data Exploration** - Explore the dataset, understand distributions, and identify patterns
2. **🧹 Data Preprocessing** - Learn about data cleaning, missing value treatment, and outlier handling
3. **⚙️ Feature Engineering** - Discover feature creation, selection, and transformation processes
4. **🤖 Model Development** - Compare different models and understand their performance
5. **🎯 House Price Predictor** - Use the final model to predict house prices interactively
""")

# Key findings summary
st.header("Key Findings Summary")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Quality")
    st.markdown("""
    - **1,460** training samples with **81** features
    - **6.6%** missing values, mostly in optional features
    - **34** data quality issues identified across **31** properties
    - **2** problematic outliers requiring removal
    """)

with col2:
    st.subheader("Model Performance")
    st.markdown("""
    - **12** different models tested and compared
    - **Ensemble methods** achieved best performance
    - **Log transformation** improved target variable normality
    - **Feature engineering** significantly enhanced predictions
    """)

# Footer
st.markdown("---")
st.markdown("""
**Team Project** | **Data Science Course** | **Ames Housing Dataset Analysis**

*This application demonstrates a complete machine learning pipeline including data exploration, 
preprocessing, feature engineering, model development, and deployment.*
""")