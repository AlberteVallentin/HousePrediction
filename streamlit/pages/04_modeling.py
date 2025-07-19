import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import json
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to access utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_processed_data
from utils.model_utils import load_model_summary, load_model_and_metadata
from utils.data_loader import load_feature_descriptions_cached

# Page configuration
st.set_page_config(
    page_title="Model Development & Optimization - Ames Housing",
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
    .insight-box {
        background-color: #e8f6f3;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Model Development & Optimization</h1>', unsafe_allow_html=True)
st.markdown("**Advanced model development, hyperparameter tuning, and performance optimization using the feature engineered dataset**")

# Load data with caching
@st.cache_data
def load_modeling_data():
    """Load model results and performance data."""
    processed_data = load_processed_data()
    model_summary = load_model_summary()
    return processed_data, model_summary

# Load model and feature descriptions
@st.cache_resource
def load_prediction_resources():
    """Load model and feature descriptions for prediction."""
    model, metadata = load_model_and_metadata("final_model")
    feature_descriptions = load_feature_descriptions_cached()
    processed_data = load_processed_data()
    return model, metadata, feature_descriptions, processed_data

with st.spinner("Loading modeling results..."):
    processed_data, model_summary = load_modeling_data()
    final_model, final_metadata, feature_descriptions, _ = load_prediction_resources()

if not processed_data or not model_summary:
    st.error("Model data not found. Please run the modeling pipeline first.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("## Navigation")
sections = {
    "Dataset Overview": "overview",
    "Baseline Evaluation": "baseline", 
    "Hyperparameter Optimization": "optimization",
    "Ensemble Methods": "ensemble",
    "Final Model Selection": "final",
    "Feature Importance Analysis": "importance",
    "Model Performance": "performance",
    "Key Insights": "insights"
}

selected_section = st.sidebar.radio("Jump to section:", list(sections.keys()))

# Helper functions for advanced feature importance analysis
@st.cache_data
def calculate_correlation_importance(X_train, y_train):
    """Calculate feature importance based on correlation with target."""
    try:
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        return correlations.to_dict()
    except Exception as e:
        st.error(f"Error calculating correlation importance: {str(e)}")
        return {}

def calculate_permutation_importance(model, X_val, y_val, top_n=20):
    """Calculate permutation importance for top features."""
    try:
        if model is None:
            return {}
        
        # Use a subset of features for faster computation
        feature_names = X_val.columns.tolist()
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_val, y_val, 
            n_repeats=5, 
            random_state=42,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Get top N features
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df.set_index('feature').to_dict()
    except Exception as e:
        st.warning(f"Permutation importance calculation failed: {str(e)}")
        return {}

def get_model_feature_importance(model):
    """Extract built-in feature importance from model."""
    try:
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        else:
            return None
    except Exception:
        return None

def create_feature_importance_plot(importance_dict, title, top_n=15):
    """Create feature importance visualization."""
    if not importance_dict:
        return None
    
    # Convert to DataFrame and get top N
    if isinstance(importance_dict, dict):
        df = pd.DataFrame(list(importance_dict.items()), columns=['feature', 'importance'])
    else:
        df = importance_dict
    
    df = df.sort_values('importance', ascending=False).head(top_n)
    
    fig = go.Figure(data=go.Bar(
        y=df['feature'],
        x=df['importance'],
        orientation='h',
        marker_color=px.colors.sequential.Blues_r
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=max(400, len(df) * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# SECTION 1: Dataset Overview
if sections[selected_section] == "overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    project_info = model_summary.get('project_info', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Features", project_info.get('features', 191))
    with col2:
        st.metric("Training Samples", f"{project_info.get('training_samples', 1458):,}")
    with col3:
        st.metric("Test Samples", f"{project_info.get('test_samples', 1459):,}")
    with col4:
        st.metric("Target Variable", "SalePrice (log-transformed)")
    
    st.markdown("---")
    
    # Data quality validation summary
    st.subheader("Data Quality Validation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Missing Values", "0", help="All missing values handled in preprocessing")
    with col2:
        st.metric("Infinite Values", "0", help="No infinite values detected")
    with col3:
        st.metric("Data Quality", "Clean", help="Ready for modeling")
    
    st.markdown("""
    <div class="success-box">
    <h4>Data Preparation Complete</h4>
    <p>191 features have been loaded across 1,458 training and 1,459 test samples. 
    Data quality validation confirms zero missing values and no infinite values, 
    indicating clean and model-ready data.</p>
    </div>
    """, unsafe_allow_html=True)

# SECTION 2: Baseline Evaluation
elif sections[selected_section] == "baseline":
    st.markdown('<h2 class="section-header">Baseline Model Evaluation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Eight baseline models were evaluated representing different algorithmic approaches:
    - **Linear Models**: Ridge, Lasso, ElasticNet (with regularization)
    - **Tree-based Models**: Random Forest, Gradient Boosting (ensemble methods)  
    - **Advanced Boosting**: XGBoost, CatBoost, LightGBM (state-of-the-art gradient boosting)
    """)
    
    # Baseline results from the actual notebook results
    baseline_data = {
        'Model': ['CatBoost', 'Ridge', 'Gradient Boosting', 'LightGBM', 'Random Forest', 'XGBoost', 'ElasticNet', 'Lasso'],
        'CV RMSE': [0.1166, 0.1218, 0.1281, 0.1309, 0.1418, 0.1419, 0.3174, 0.3203],
        'CV Std': [0.0099, 0.0089, 0.0113, 0.0088, 0.0095, 0.0082, 0.0248, 0.0253],
        'Algorithm Type': ['Gradient Boosting', 'Linear', 'Gradient Boosting', 'Gradient Boosting', 
                          'Tree Ensemble', 'Gradient Boosting', 'Linear', 'Linear']
    }
    
    baseline_df = pd.DataFrame(baseline_data)
    
    # Display results table
    st.subheader("Baseline Performance Results")
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)
    
    # Baseline performance visualization
    fig = px.bar(baseline_df, 
                 x='Model', 
                 y='CV RMSE',
                 color='Algorithm Type',
                 title='Baseline Model Performance (5-Fold Cross-Validation RMSE)',
                 error_y='CV Std')
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("""
    <div class="insight-box">
    <h4>Key Baseline Insights</h4>
    <ul>
        <li><strong>Best Performer:</strong> CatBoost (RMSE: 0.1166) - excellent handling of categorical features</li>
        <li><strong>Algorithm Ranking:</strong> Gradient Boosting > Linear > Tree Ensemble methods</li>
        <li><strong>Linear Models:</strong> Ridge outperforms Lasso/ElasticNet significantly</li>
        <li><strong>Consistency:</strong> Most models show low standard deviation, indicating stable performance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 3: Hyperparameter Optimization
elif sections[selected_section] == "optimization":
    st.markdown('<h2 class="section-header">Hyperparameter Optimization</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Optuna framework was used for advanced hyperparameter optimization with 415 total trials 
    across all models. Each model underwent systematic parameter tuning using Tree-structured 
    Parzen Estimator (TPE) sampling.
    """)
    
    # Optimization results comparison
    if 'individual_models' in model_summary:
        optimization_data = []
        baseline_rmse = {
            'Ridge': 0.1218, 'Lasso': 0.3203, 'ElasticNet': 0.3174,
            'RandomForest': 0.1418, 'GradientBoosting': 0.1281,
            'XGBoost': 0.1419, 'CatBoost': 0.1166, 'LightGBM': 0.1309
        }
        
        for model_name, model_info in model_summary['individual_models'].items():
            model_base_name = model_name.replace('_Optimized', '')
            baseline_score = baseline_rmse.get(model_base_name, 0)
            optimized_score = model_info['cv_rmse_mean']
            improvement = ((baseline_score - optimized_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            optimization_data.append({
                'Model': model_base_name,
                'Baseline RMSE': baseline_score,
                'Optimized RMSE': optimized_score,
                'Improvement (%)': improvement,
                'CV Std': model_info['cv_rmse_std']
            })
        
        opt_df = pd.DataFrame(optimization_data)
        opt_df = opt_df.sort_values('Optimized RMSE')
        
        st.subheader("Optimization Results")
        st.dataframe(opt_df, use_container_width=True, hide_index=True)
        
        # Improvement visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=opt_df['Model'],
            y=opt_df['Baseline RMSE'],
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            name='Optimized',
            x=opt_df['Model'],
            y=opt_df['Optimized RMSE'],
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Baseline vs Optimized Performance',
            xaxis_title='Model',
            yaxis_title='CV RMSE',
            barmode='group',
            height=500
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_trials = 415
            st.metric("Total Optimization Trials", f"{total_trials:,}")
        with col2:
            avg_improvement = opt_df['Improvement (%)'].mean()
            st.metric("Average Improvement", f"{avg_improvement:.2f}%")
        with col3:
            best_improvement = opt_df['Improvement (%)'].max()
            best_model = opt_df.loc[opt_df['Improvement (%)'].idxmax(), 'Model']
            st.metric("Best Improvement", f"{best_improvement:.2f}% ({best_model})")

# SECTION 4: Ensemble Methods
elif sections[selected_section] == "ensemble":
    st.markdown('<h2 class="section-header">Ensemble Methods</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Three ensemble strategies were implemented to combine the top 4 performing models:
    CatBoost, XGBoost, LightGBM, and Lasso. Each approach uses different combination techniques.
    """)
    
    # Ensemble component models
    st.subheader("Ensemble Component Models")
    component_data = {
        'Model': ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso'],
        'Individual CV RMSE': [0.1143, 0.1148, 0.1172, 0.1175],
        'Selection Criteria': ['Best CV Performance', '2nd Best Performance', '3rd Best Performance', '4th Best Performance'],
        'Algorithm Type': ['Gradient Boosting', 'Gradient Boosting', 'Gradient Boosting', 'Linear Regularized']
    }
    
    comp_df = pd.DataFrame(component_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Ensemble results
    if 'ensemble_models' in model_summary:
        st.subheader("Ensemble Performance Results")
        
        ensemble_data = []
        for ensemble_name, ensemble_info in model_summary['ensemble_models'].items():
            ensemble_data.append({
                'Ensemble Method': ensemble_name.replace('_', ' '),
                'CV RMSE': ensemble_info['cv_rmse_mean'],
                'CV Std': ensemble_info['cv_rmse_std'],
                'Components': ', '.join(ensemble_info['components'])
            })
        
        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_df = ensemble_df.sort_values('CV RMSE')
        st.dataframe(ensemble_df, use_container_width=True, hide_index=True)
        
        # Ensemble vs individual comparison
        fig = go.Figure()
        
        # Add individual models
        individual_models = ['CatBoost', 'XGBoost', 'LightGBM', 'Lasso']
        individual_scores = [0.1143, 0.1148, 0.1172, 0.1175]
        
        fig.add_trace(go.Bar(
            name='Individual Models',
            x=individual_models,
            y=individual_scores,
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add ensemble models
        ensemble_names = [name.replace('_', ' ') for name in model_summary['ensemble_models'].keys()]
        ensemble_scores = [info['cv_rmse_mean'] for info in model_summary['ensemble_models'].values()]
        
        fig.add_trace(go.Bar(
            name='Ensemble Models',
            x=ensemble_names,
            y=ensemble_scores,
            marker_color='darkgreen',
            opacity=0.8
        ))
        
        fig.update_layout(
            title='Individual vs Ensemble Model Performance',
            xaxis_title='Model',
            yaxis_title='CV RMSE',
            barmode='group',
            height=500
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Ensemble insights
        best_ensemble = min(ensemble_df['CV RMSE'])
        best_individual = min(individual_scores)
        ensemble_improvement = ((best_individual - best_ensemble) / best_individual * 100)
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>Ensemble Method Insights</h4>
        <ul>
            <li><strong>Best Ensemble:</strong> {ensemble_df.iloc[0]['Ensemble Method']} (RMSE: {best_ensemble:.4f})</li>
            <li><strong>Improvement over Best Individual:</strong> {ensemble_improvement:.2f}%</li>
            <li><strong>Ensemble Advantage:</strong> Combines diverse model strengths and reduces overfitting</li>
            <li><strong>Stacking Performance:</strong> Meta-model learns optimal combination weights</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# SECTION 5: Final Model Selection
elif sections[selected_section] == "final":
    st.markdown('<h2 class="section-header">Final Model Selection</h2>', unsafe_allow_html=True)
    
    # Final model information
    final_model_info = model_summary.get('final_model', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Model", final_model_info.get('final_model_name', 'Stacking Ensemble'))
    with col2:
        st.metric("CV RMSE", f"{final_model_info.get('final_cv_rmse', 0.1114):.4f}")
    with col3:
        st.metric("Selection Criteria", "Best Cross-Validation Performance")
    
    # Performance comparison of all models
    st.subheader("Complete Model Performance Ranking")
    
    if 'performance_comparison' in model_summary:
        performance_data = []
        for model_name, rmse in model_summary['performance_comparison'].items():
            model_type = "Ensemble" if "Ensemble" in model_name else "Individual"
            performance_data.append({
                'Rank': 0,  # Will be filled after sorting
                'Model': model_name.replace('_', ' '),
                'CV RMSE': rmse,
                'Type': model_type,
                'Final Selection': model_name == final_model_info.get('final_model_name', '')
            })
        
        perf_df = pd.DataFrame(performance_data).sort_values('CV RMSE')
        perf_df['Rank'] = range(1, len(perf_df) + 1)
        
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Performance visualization with highlighting
        colors = ['gold' if final else ('green' if type_val == 'Ensemble' else 'blue') 
                 for final, type_val in zip(perf_df['Final Selection'], perf_df['Type'])]
        
        fig = go.Figure(data=go.Bar(
            x=perf_df['Model'],
            y=perf_df['CV RMSE'],
            marker_color=colors,
            text=[f"{rmse:.4f}" for rmse in perf_df['CV RMSE']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Final Model Performance Comparison (Final Model in Gold)',
            xaxis_title='Model',
            yaxis_title='Cross-Validation RMSE',
            height=500
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance progression analysis
        baseline_best = 0.1166  # CatBoost baseline
        final_rmse = final_model_info.get('final_cv_rmse', 0.1114)
        total_improvement = ((baseline_best - final_rmse) / baseline_best * 100)
        
        st.markdown(f"""
        <div class="success-box">
        <h4>Final Model Achievement</h4>
        <p>The <strong>Stacking Ensemble</strong> achieved the best overall cross-validation performance with 
        an RMSE of <strong>0.1114</strong> and was selected as the final model. It marginally outperformed 
        other ensemble strategies and all individually optimized models.</p>
        <p><strong>Total Improvement:</strong> {total_improvement:.2f}% over best baseline model</p>
        </div>
        """, unsafe_allow_html=True)

# SECTION 6: Feature Importance Analysis
elif sections[selected_section] == "importance":
    st.markdown('<h2 class="section-header">Feature Importance Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive feature importance analysis using multiple methods to understand which 
    features drive house price predictions most effectively.
    """)
    
    # Load processed data for feature importance analysis
    if 'X_train' in processed_data and 'y_train' in processed_data:
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Handle different y_train formats
        if isinstance(y_train, pd.DataFrame):
            if 'SalePrice' in y_train.columns:
                y_train = y_train['SalePrice']
            else:
                y_train = y_train.iloc[:, 0]  # Take first column
        
        
        # Create validation split for permutation importance
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Method selection
        importance_method = st.selectbox(
            "Select Feature Importance Method:",
            ["Correlation Analysis", "Permutation Importance"]
        )
        
        if importance_method == "Correlation Analysis":
            st.subheader("Correlation-Based Feature Importance")
            st.markdown("Features ranked by absolute correlation with target variable (log SalePrice)")
            
            with st.spinner("Calculating correlation importance..."):
                corr_importance = calculate_correlation_importance(X_train, y_train)
            
            if corr_importance:
                corr_fig = create_feature_importance_plot(
                    corr_importance, 
                    "Top 20 Features by Correlation with Target",
                    top_n=20
                )
                if corr_fig:
                    st.plotly_chart(corr_fig, use_container_width=True)
                
                # Top correlation features table
                corr_df = pd.DataFrame(list(corr_importance.items()), columns=['Feature', 'Correlation'])
                corr_df = corr_df.sort_values('Correlation', ascending=False).head(15)
                
                st.subheader("Top 15 Correlated Features")
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
                
                # Correlation insights
                positive_corr = sum(1 for v in corr_importance.values() if v > 0)
                strong_corr = sum(1 for v in corr_importance.values() if abs(v) > 0.3)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features", len(corr_importance))
                with col2:
                    st.metric("Strong Correlations (|r| > 0.3)", strong_corr)
                with col3:
                    top_feature = corr_df.iloc[0]['Feature']
                    top_corr = corr_df.iloc[0]['Correlation']
                    st.metric("Strongest Predictor", f"{top_feature} ({top_corr:.3f})")
        
        elif importance_method == "Permutation Importance":
            st.subheader("Permutation Feature Importance")
            st.markdown("Importance calculated by measuring performance drop when features are randomly shuffled")
            
            if final_model is not None:
                with st.spinner("Calculating permutation importance (this may take a moment)..."):
                    perm_importance = calculate_permutation_importance(
                        final_model, X_val_split, y_val_split, top_n=20
                    )
                
                if perm_importance and 'importance' in perm_importance:
                    perm_fig = create_feature_importance_plot(
                        perm_importance['importance'],
                        "Top 20 Features by Permutation Importance",
                        top_n=20
                    )
                    if perm_fig:
                        st.plotly_chart(perm_fig, use_container_width=True)
                    
                    # Permutation importance table with std
                    perm_data = []
                    for feature in list(perm_importance['importance'].keys())[:15]:
                        perm_data.append({
                            'Feature': feature,
                            'Importance': perm_importance['importance'][feature],
                            'Std': perm_importance.get('std', {}).get(feature, 0)
                        })
                    
                    perm_df = pd.DataFrame(perm_data)
                    
                    st.subheader("Top 15 Features with Standard Deviation")
                    st.dataframe(perm_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("""
                    <div class="insight-box">
                    <h4>Permutation Importance Insights</h4>
                    <p>Permutation importance measures the increase in model error when feature values 
                    are randomly shuffled. Higher values indicate more important features for model performance.</p>
                    <p>Standard deviation shows the stability of importance across different random shuffles.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Permutation importance calculation failed or returned no results.")
            else:
                st.error("Final model not loaded. Cannot calculate permutation importance.")
    else:
        st.error("Training data not available for feature importance analysis.")
        st.info("Available keys in processed_data: " + str(list(processed_data.keys()) if processed_data else "None"))
        
        # Try alternative data loading
        st.markdown("### Attempting Alternative Data Loading")
        try:
            # Try to load directly from files
            import os
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
            X_train_path = os.path.join(data_path, 'X_train_final.csv')
            y_train_path = os.path.join(data_path, 'y_train_final.csv')
            
            if os.path.exists(X_train_path) and os.path.exists(y_train_path):
                X_train = pd.read_csv(X_train_path)
                y_train_df = pd.read_csv(y_train_path)
                y_train = y_train_df['SalePrice'] if 'SalePrice' in y_train_df.columns else y_train_df.iloc[:, 0]
                
                
                # Create validation split
                from sklearn.model_selection import train_test_split
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                # Continue with feature importance analysis
                feature_importance_analysis_available = True
            else:
                st.error("Could not find X_train_final.csv or y_train_final.csv files")
                feature_importance_analysis_available = False
        except Exception as e:
            st.error(f"Error loading data directly: {str(e)}")
            feature_importance_analysis_available = False
        
        if not feature_importance_analysis_available:
            st.warning("Feature importance analysis is not available without training data.")
            

# SECTION 7: Model Performance
elif sections[selected_section] == "performance":
    st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive performance analysis of the final model including validation metrics,
    residual analysis, and prediction quality assessment.
    """)
    
    # Final model validation metrics
    st.subheader("Final Model Validation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE (Original Scale)", "$18,713", help="Root Mean Square Error in dollars")
    with col2:
        st.metric("MAE (Original Scale)", "$13,499", help="Mean Absolute Error in dollars")
    with col3:
        st.metric("R² Score", "0.9366", help="Coefficient of determination")
    with col4:
        st.metric("MAPE", "8.18%", help="Mean Absolute Percentage Error")
    
    # Performance progression visualization
    st.subheader("Performance Progression Analysis")
    
    progression_data = {
        'Stage': ['Best Baseline', 'Best Individual', 'Final Ensemble'],
        'CV RMSE': [0.1166, 0.1143, 0.1114],
        'Improvement (%)': [0, 2.00, 4.51]
    }
    
    prog_df = pd.DataFrame(progression_data)
    
    # Create progression chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prog_df['Stage'],
        y=prog_df['CV RMSE'],
        mode='lines+markers',
        marker=dict(size=12, color='blue'),
        line=dict(width=3),
        name='CV RMSE'
    ))
    
    fig.add_trace(go.Bar(
        x=prog_df['Stage'],
        y=prog_df['Improvement (%)'],
        name='Improvement %',
        yaxis='y2',
        marker_color='lightgreen',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Model Development Performance Progression',
        xaxis_title='Development Stage',
        yaxis_title='Cross-Validation RMSE',
        yaxis2=dict(
            title='Improvement (%)',
            overlaying='y',
            side='right'
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model development statistics
    st.subheader("Development Statistics Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Process Metrics**")
        stats_data = {
            'Metric': ['Baseline Models Tested', 'Hyperparameter Trials', 'Ensemble Methods', 'Total Models Evaluated'],
            'Count': [8, 415, 3, 11]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Performance Metrics**")
        perf_data = {
            'Metric': ['Best Baseline RMSE', 'Final Model RMSE', 'Total Improvement', 'Validation R²'],
            'Value': ['0.1166', '0.1114', '+4.51%', '0.9366']
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Submission details
    st.subheader("Model Deployment Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predictions Generated", "1,459", help="Test set predictions")
    with col2:
        st.metric("Price Range", "$47,690 - $701,555", help="Prediction range")
    with col3:
        st.metric("Submission File", "submission_Stacking_Ensemble6.csv")
    
    st.markdown("""
    <div class="success-box">
    <h4>Model Performance Summary</h4>
    <p>The final stacking ensemble model achieved excellent performance with:</p>
    <ul>
        <li><strong>High Accuracy:</strong> R² of 0.9366 explains 93.66% of price variance</li>
        <li><strong>Low Error:</strong> RMSE of $18,713 represents ~8.18% average error</li>
        <li><strong>Realistic Predictions:</strong> Price range covers expected housing market spectrum</li>
        <li><strong>Robust Generalization:</strong> Consistent performance across validation sets</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# SECTION 8: Key Insights
elif sections[selected_section] == "insights":
    st.markdown('<h2 class="section-header">Key Insights & Learnings</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Comprehensive analysis of the modeling pipeline reveals important insights about 
    algorithm performance, ensemble effectiveness, and optimization strategies.
    """)
    
    # Algorithm performance insights
    st.subheader("Algorithm Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Best Performing Algorithm Types:**
        1. **Gradient Boosting Models** (CatBoost, XGBoost, LightGBM)
           - Superior handling of complex feature interactions
           - Excellent performance on tabular data
           - Built-in regularization prevents overfitting
        
        2. **Regularized Linear Models** (Lasso, Ridge)
           - Strong baseline performance after optimization
           - Good interpretability and fast training
           - Effective with engineered features
        
        3. **Traditional Ensemble Methods** (Random Forest)
           - Decent performance but outpaced by modern boosting
           - Higher variance in cross-validation
        """)
    
    with col2:
        # Algorithm type performance chart
        algo_data = {
            'Algorithm Type': ['Gradient Boosting', 'Linear Regularized', 'Tree Ensemble'],
            'Best RMSE': [0.1143, 0.1175, 0.1412],
            'Count': [4, 3, 1]
        }
        
        algo_df = pd.DataFrame(algo_data)
        
        fig = px.bar(algo_df, 
                     x='Algorithm Type', 
                     y='Best RMSE',
                     title='Best Performance by Algorithm Type',
                     color='Best RMSE',
                     color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimization insights
    st.subheader("Hyperparameter Optimization Insights")
    
    st.markdown("""
    <div class="insight-box">
    <h4>Optimization Strategy Effectiveness</h4>
    <ul>
        <li><strong>High Impact Models:</strong> Linear models showed dramatic improvement (ElasticNet: 63% boost)</li>
        <li><strong>Diminishing Returns:</strong> Tree-based models had smaller but consistent gains</li>
        <li><strong>Optimal Trial Count:</strong> 50-100 trials per model provided best efficiency/performance balance</li>
        <li><strong>TPE Sampling:</strong> Outperformed random search in convergence speed</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensemble insights
    st.subheader("Ensemble Method Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Ensemble Strategy Results:**
        - **Stacking Ensemble:** Best performance (0.1114 RMSE)
        - **Simple/Weighted Averaging:** Similar results (0.1117 RMSE)
        - **Meta-model Learning:** Stacking's ridge meta-model optimally combines predictions
        - **Diversity Benefit:** Combining different algorithm types (boosting + linear) improves robustness
        """)
    
    with col2:
        # Ensemble comparison
        ens_data = {
            'Method': ['Stacking', 'Weighted', 'Simple'],
            'RMSE': [0.1114, 0.1117, 0.1117],
            'Complexity': ['High', 'Medium', 'Low']
        }
        
        ens_df = pd.DataFrame(ens_data)
        
        fig = px.scatter(ens_df, 
                        x='Complexity', 
                        y='RMSE',
                        size=[100, 80, 60],
                        color='Method',
                        title='Ensemble Method Trade-offs')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature engineering insights
    st.subheader("Feature Engineering Impact")
    
    st.markdown("""
    **Key Feature Engineering Contributions:**
    - **Feature Count:** 191 engineered features from 81 original features
    - **Categorical Encoding:** One-hot encoding created interpretable binary features
    - **Feature Creation:** Age calculations, area combinations, and ratios added predictive power
    - **Skewness Treatment:** Log transformations improved linear model performance
    """)
    
    # Final recommendations
    st.subheader("Recommendations for Future Work")
    
    st.markdown("""
    <div class="insight-box">
    <h4>Model Improvement Strategies</h4>
    <ol>
        <li><strong>Advanced Ensembling:</strong> Explore neural network meta-learners for stacking</li>
        <li><strong>Feature Selection:</strong> Use recursive feature elimination to reduce overfitting</li>
        <li><strong>Cross-Validation:</strong> Implement stratified CV based on price ranges</li>
        <li><strong>External Data:</strong> Incorporate economic indicators and market trends</li>
        <li><strong>Model Interpretability:</strong> Add SHAP analysis for production deployment</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    <div class="success-box">
    <h4>Project Conclusion</h4>
    <p>The modeling pipeline successfully delivered a high-performing stacking ensemble that 
    <strong>exceeds all target metrics</strong>. The systematic approach of baseline evaluation, 
    hyperparameter optimization, and ensemble methods resulted in a <strong>4.51% improvement</strong> 
    over baseline performance.</p>
    <p>The final model demonstrates excellent generalization with an R² of 0.9366 and realistic 
    prediction ranges, making it suitable for production deployment in house price estimation applications.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This analysis follows the methodology from notebook 04_modeling.ipynb*")