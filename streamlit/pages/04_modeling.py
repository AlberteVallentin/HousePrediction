import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json

# Add the parent directory to access utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_processed_data
from utils.model_utils import load_model_summary
from utils.visualizations import create_model_comparison_plot

# Page configuration
st.set_page_config(
    page_title="Model Development",
    layout="wide"
)

st.title("Model Development")
st.markdown("---")

st.markdown("""
This page demonstrates the comprehensive model development process including baseline evaluation, 
hyperparameter optimization, ensemble methods, and final model selection.
""")

# Load data
@st.cache_data
def load_modeling_data():
    """Load model results and performance data."""
    processed_data = load_processed_data()
    model_summary = load_model_summary()
    return processed_data, model_summary

processed_data, model_summary = load_modeling_data()

if not processed_data or not model_summary:
    st.error("Model data not found. Please run the modeling pipeline first.")
    st.stop()

# Section 1: Dataset Overview
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Features", model_summary.get('project_info', {}).get('features', 'N/A'))
with col2:
    st.metric("Training Samples", model_summary.get('project_info', {}).get('training_samples', 'N/A'))
with col3:
    st.metric("Test Samples", model_summary.get('project_info', {}).get('test_samples', 'N/A'))

# Section 2: Baseline Model Evaluation
st.header("2. Baseline Model Evaluation")

st.markdown("""
Eight baseline models were evaluated representing different algorithmic approaches:
- **Linear Models**: Ridge, Lasso, ElasticNet
- **Tree-based Models**: Random Forest, Gradient Boosting
- **Advanced Boosting**: XGBoost, CatBoost, LightGBM
""")

# Baseline results table
baseline_data = {
    'Model': ['CatBoost', 'Ridge', 'Gradient Boosting', 'LightGBM', 'Random Forest', 'XGBoost', 'ElasticNet', 'Lasso'],
    'CV RMSE': [0.1166, 0.1218, 0.1281, 0.1309, 0.1418, 0.1419, 0.3174, 0.3203],
    'CV Std': [0.0099, 0.0089, 0.0113, 0.0088, 0.0095, 0.0082, 0.0248, 0.0253]
}

baseline_df = pd.DataFrame(baseline_data)
st.dataframe(baseline_df, use_container_width=True)

# Baseline performance visualization
fig = px.bar(baseline_df, x='Model', y='CV RMSE', 
             title='Baseline Model Performance (Cross-Validation RMSE)',
             error_y='CV Std')
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Section 3: Hyperparameter Optimization
st.header("3. Hyperparameter Optimization")

st.markdown("""
Each model underwent hyperparameter optimization using Optuna's Bayesian optimization:
- **Total Trials**: 415 optimization trials across all models
- **Optimization Strategy**: Model-specific parameter spaces and trial counts
- **Objective Function**: Minimize cross-validated RMSE
""")

# Optimization results comparison
if 'individual_models' in model_summary:
    optimization_data = []
    baseline_rmse = {'Ridge': 0.1218, 'Lasso': 0.3203, 'ElasticNet': 0.3174, 
                     'RandomForest': 0.1418, 'GradientBoosting': 0.1281, 
                     'XGBoost': 0.1419, 'CatBoost': 0.1166, 'LightGBM': 0.1309}
    
    for model_name, model_data in model_summary['individual_models'].items():
        model_base = model_name.replace('_Optimized', '')
        baseline_score = baseline_rmse.get(model_base, 0)
        optimized_score = model_data['cv_rmse_mean']
        improvement = baseline_score - optimized_score
        
        optimization_data.append({
            'Model': model_base,
            'Baseline RMSE': baseline_score,
            'Optimized RMSE': optimized_score,
            'Improvement': improvement
        })
    
    opt_df = pd.DataFrame(optimization_data)
    st.dataframe(opt_df, use_container_width=True)
    
    # Show biggest improvements
    st.subheader("Optimization Impact")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Biggest Improvements:**")
        top_improvements = opt_df.nlargest(3, 'Improvement')
        for _, row in top_improvements.iterrows():
            st.text(f"• {row['Model']}: {row['Improvement']:.4f}")
    
    with col2:
        st.markdown("**Optimization Statistics:**")
        st.text(f"• Models improved: {(opt_df['Improvement'] > 0).sum()}/8")
        st.text(f"• Average improvement: {opt_df['Improvement'].mean():.4f}")
        st.text(f"• Max improvement: {opt_df['Improvement'].max():.4f}")

# Section 4: Ensemble Methods
st.header("4. Ensemble Methods")

st.markdown("""
Three ensemble strategies were implemented to combine the top 4 performing models:
""")

# Ensemble component models
ensemble_components = {
    'Component Models': ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso'],
    'Individual RMSE': [0.1143, 0.1148, 0.1172, 0.1175],
    'Selection Criteria': ['Best CV RMSE', '2nd Best CV RMSE', '3rd Best CV RMSE', '4th Best CV RMSE']
}

comp_df = pd.DataFrame(ensemble_components)
st.dataframe(comp_df, use_container_width=True)

# Ensemble results
if 'ensemble_models' in model_summary:
    ensemble_data = []
    for ensemble_name, ensemble_info in model_summary['ensemble_models'].items():
        ensemble_data.append({
            'Ensemble Type': ensemble_name.replace('_', ' '),
            'CV RMSE': ensemble_info['cv_rmse_mean'],
            'CV Std': ensemble_info['cv_rmse_std']
        })
    
    ensemble_df = pd.DataFrame(ensemble_data)
    st.dataframe(ensemble_df, use_container_width=True)
    
    # Ensemble performance comparison
    fig = px.bar(ensemble_df, x='Ensemble Type', y='CV RMSE',
                 title='Ensemble Method Performance',
                 error_y='CV Std')
    st.plotly_chart(fig, use_container_width=True)

# Section 5: Final Model Selection
st.header("5. Final Model Selection")

st.markdown("""
The final model was selected based on cross-validation performance:
""")

# Performance comparison of all models
if 'performance_comparison' in model_summary:
    performance_data = []
    for model_name, rmse in model_summary['performance_comparison'].items():
        model_type = "Ensemble" if "Ensemble" in model_name else "Individual"
        performance_data.append({
            'Model': model_name.replace('_', ' '),
            'CV RMSE': rmse,
            'Type': model_type
        })
    
    perf_df = pd.DataFrame(performance_data).sort_values('CV RMSE')
    
    # Highlight final model
    final_model_name = model_summary.get('final_model', {}).get('final_model_name', 'Unknown')
    perf_df['Final Model'] = perf_df['Model'] == final_model_name.replace('_', ' ')
    
    st.dataframe(perf_df, use_container_width=True)
    
    # Performance visualization
    fig = px.bar(perf_df, x='Model', y='CV RMSE', 
                 color='Type',
                 title='Final Model Performance Comparison')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Section 6: Final Model Performance
st.header("6. Final Model Performance")

final_model_info = model_summary.get('final_model', {})
if final_model_info:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Model", final_model_info.get('final_model_name', 'Unknown'))
    
    with col2:
        st.metric("CV RMSE", f"{final_model_info.get('final_cv_rmse', 0):.4f}")
    
    with col3:
        st.metric("Selection Criteria", "Best CV Performance")

# Section 7: Model Development Statistics
st.header("7. Model Development Statistics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Development Process")
    st.metric("Baseline Models", "8")
    st.metric("Optimization Trials", "415")
    st.metric("Ensemble Methods", "3")
    st.metric("Total Models Evaluated", "11")

with col2:
    st.subheader("Performance Improvements")
    if 'individual_models' in model_summary and 'ensemble_models' in model_summary:
        best_individual = min(model_summary['individual_models'].values(), 
                             key=lambda x: x['cv_rmse_mean'])['cv_rmse_mean']
        best_ensemble = min(model_summary['ensemble_models'].values(),
                           key=lambda x: x['cv_rmse_mean'])['cv_rmse_mean']
        
        baseline_best = 0.1166  # CatBoost baseline
        individual_improvement = ((baseline_best - best_individual) / baseline_best) * 100
        ensemble_improvement = ((baseline_best - best_ensemble) / baseline_best) * 100
        
        st.metric("Individual Optimization", f"{individual_improvement:.2f}%")
        st.metric("Ensemble Improvement", f"{ensemble_improvement:.2f}%")
        st.metric("Total Improvement", f"{ensemble_improvement:.2f}%")

# Section 8: Interactive Model Performance Dashboard
st.header("8. Interactive Model Performance Dashboard")

st.markdown("""
Comprehensive dashboard for analyzing model performance across different metrics and perspectives.
""")

# Create tabs for different performance analyses
tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Optimization Analysis", "Ensemble Insights", "Model Comparison"])

with tab1:
    st.subheader("Comprehensive Performance Metrics")
    
    st.info("Detailed performance metrics would be loaded from the actual model training results.")

with tab2:
    st.subheader("Hyperparameter Optimization Analysis")
    
    st.info("Hyperparameter optimization analysis would show the actual optimization results from the model training pipeline.")

with tab3:
    st.subheader("Ensemble Method Analysis")
    
    st.info("Ensemble analysis would show the actual ensemble performance and component weights from the trained models.")

with tab4:
    st.subheader("Model Comparison Matrix")
    
    st.info("Model comparison analysis would show actual performance metrics across different dimensions based on the trained models.")

# Section 9: Key Insights
st.header("9. Key Insights")

st.markdown("""
**Model Performance Hierarchy:**
1. **Ensemble Methods** consistently outperformed individual models
2. **Tree-based Models** (CatBoost, XGBoost, LightGBM) showed strong baseline performance
3. **Linear Models** benefited significantly from hyperparameter optimization

**Optimization Impact:**
- **Lasso and ElasticNet** showed the largest improvements (~0.20 RMSE reduction)
- **Tree-based models** had smaller but consistent improvements
- **Ridge regression** was already near-optimal at baseline

**Ensemble Effectiveness:**
- **Stacking Ensemble** achieved the best performance (RMSE: 0.1114)
- **Simple averaging** performed nearly as well as sophisticated weighting
- **Model diversity** contributed to ensemble success

**Algorithm Insights:**
- **CatBoost** emerged as the strongest individual model
- **Gradient boosting** variants consistently outperformed random forests
- **Regularization** was crucial for linear model performance

**Model Development Process:**
- Systematic hyperparameter optimization provided measurable improvements
- Ensemble methods offered the final performance edge
- Cross-validation provided reliable model selection guidance

The final stacking ensemble achieved a 4.51% improvement over the best baseline model, 
demonstrating the value of a comprehensive modeling approach.
""")

# Footer
st.markdown("---")
st.markdown("*This modeling pipeline represents a comprehensive approach to machine learning model development, from baseline evaluation through advanced ensemble methods.*")