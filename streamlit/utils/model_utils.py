import joblib
import pandas as pd
import numpy as np
import streamlit as st
import os
import json
from typing import Dict, Any, Tuple

@st.cache_resource
def load_model_and_metadata(model_name="final_model"):
    """Load trained model and its metadata."""
    try:
        models_path = os.path.join(os.path.dirname(__file__), '../../models/')
        
        # Load model
        model_path = os.path.join(models_path, f"{model_name}.joblib")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            return None, None
        
        # Load metadata
        metadata_path = os.path.join(models_path, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            st.warning(f"Metadata file not found: {metadata_path}")
            metadata = {}
        
        return model, metadata
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_model_summary():
    """Load model summary with performance metrics."""
    try:
        summary_path = os.path.join(os.path.dirname(__file__), '../../models/model_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Error loading model summary: {str(e)}")
        return {}

@st.cache_data
def load_feature_order():
    """Load the correct feature order for the trained model."""
    try:
        feature_order_path = os.path.join(os.path.dirname(__file__), '../../models/feature_order.json')
        if os.path.exists(feature_order_path):
            with open(feature_order_path, 'r') as f:
                return json.load(f)['feature_names']
        else:
            st.error("Feature order file not found. Please ensure the model training process saves feature order.")
            return []
    except Exception as e:
        st.error(f"Error loading feature order: {str(e)}")
        return []

def get_available_models():
    """Get list of available trained models."""
    models_path = os.path.join(os.path.dirname(__file__), '../../models/')
    
    if not os.path.exists(models_path):
        return []
    
    model_files = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
    model_names = [f.replace('.joblib', '') for f in model_files]
    
    return model_names

def predict_house_price(model, features_dict, metadata=None):
    """Make prediction using the loaded model."""
    try:
        # Load the correct feature order
        feature_order = load_feature_order()
        
        if not feature_order:
            st.error("Could not load feature order. Prediction cannot proceed.")
            return None
        
        # Ensure features are in the correct order
        ordered_features = {}
        for feature_name in feature_order:
            if feature_name in features_dict:
                ordered_features[feature_name] = features_dict[feature_name]
            else:
                # Set default values for missing features
                ordered_features[feature_name] = 0
        
        # Convert features to DataFrame in the correct order
        features_df = pd.DataFrame([ordered_features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Model was trained on log-transformed target, so transform back
        prediction = np.expm1(prediction)
        
        return prediction
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_feature_importance(model, feature_names):
    """Extract feature importance from model if available."""
    try:
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, abs(model.coef_)))
        else:
            return {}
    except Exception as e:
        st.error(f"Error extracting feature importance: {str(e)}")
        return {}

def format_prediction_result(prediction, confidence_interval=None):
    """Format prediction result for display."""
    if prediction is None:
        return "Prediction failed"
    
    # Format as currency
    formatted_price = f"${prediction:,.0f}"
    
    if confidence_interval:
        lower, upper = confidence_interval
        formatted_price += f" (95% CI: ${lower:,.0f} - ${upper:,.0f})"
    
    return formatted_price

def calculate_price_range(prediction, uncertainty_factor=0.1):
    """Calculate a reasonable price range around the prediction."""
    if prediction is None:
        return None, None
    
    margin = prediction * uncertainty_factor
    lower_bound = max(0, prediction - margin)
    upper_bound = prediction + margin
    
    return lower_bound, upper_bound

def validate_input_features(features_dict, required_features):
    """Validate that all required features are present."""
    missing_features = []
    
    for feature in required_features:
        if feature not in features_dict:
            missing_features.append(feature)
    
    if missing_features:
        st.error(f"Missing required features: {missing_features}")
        return False
    
    return True

def get_model_performance_metrics(model_name):
    """Get performance metrics for a specific model."""
    summary = load_model_summary()
    
    if model_name in summary:
        return summary[model_name]
    else:
        return {}

def prepare_features_for_prediction(user_inputs, feature_columns):
    """Prepare user inputs for model prediction."""
    # Initialize features with default values
    features = {}
    
    # Set user inputs
    for feature, value in user_inputs.items():
        features[feature] = value
    
    # Add missing features with reasonable defaults
    for feature in feature_columns:
        if feature not in features:
            features[feature] = 0  # Default value
    
    return features

def explain_prediction(model, features_df, feature_names, top_n=10):
    """Provide explanation for the prediction."""
    try:
        # Get feature importance
        importance = get_feature_importance(model, feature_names)
        
        if not importance:
            return "Feature importance not available for this model type."
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N features
        top_features = sorted_importance[:top_n]
        
        explanation = "**Top factors influencing the prediction:**\n\n"
        
        for i, (feature, importance_score) in enumerate(top_features, 1):
            feature_value = features_df[feature].iloc[0] if feature in features_df.columns else "N/A"
            explanation += f"{i}. **{feature}**: {feature_value} (importance: {importance_score:.3f})\n"
        
        return explanation
    
    except Exception as e:
        return f"Error generating explanation: {str(e)}"