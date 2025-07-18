import requests
import json
import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any, Optional

class OllamaChat:
    """Enhanced Ollama chat integration with actual project data access."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.project_context = self._load_project_context()
        self.system_prompt = self._create_system_prompt()
    
    def _load_project_context(self) -> Dict[str, Any]:
        """Load all available project context from files."""
        context = {
            "dataset_info": {},
            "model_results": {},
            "feature_descriptions": "",
            "notebook_summary": "",
            "data_quality_issues": []
        }
        
        # Since AI chat utils is in streamlit/utils/, we need to go up to project root
        base_path = os.path.join(os.path.dirname(__file__), '../..')
        
        # Load dataset info
        context["dataset_info"] = self._load_dataset_info(base_path)
        
        # Load model results
        context["model_results"] = self._load_model_results(base_path)
        
        # Load feature descriptions
        context["feature_descriptions"] = self._load_feature_descriptions(base_path)
        
        # Load notebook summary
        context["notebook_summary"] = self._load_notebook_summary(base_path)
        
        # Load data quality issues
        context["data_quality_issues"] = self._load_data_quality_issues(base_path)
        
        return context
    
    def _load_dataset_info(self, base_path: str) -> Dict[str, Any]:
        """Load basic dataset information from project files."""
        # Try the correct paths based on project structure
        possible_paths = [
            'data/raw/train.csv',  # Main project structure
            'data/train.csv',      # Alternative in data folder
        ]
        
        for path_suffix in possible_paths:
            try:
                train_path = os.path.join(base_path, path_suffix)
                if os.path.exists(train_path):
                    df = pd.read_csv(train_path)
                    return {
                        "training_samples": len(df),
                        "original_features": df.shape[1],
                        "target_column": "SalePrice",
                        "price_range": {
                            "min": int(df['SalePrice'].min()),
                            "max": int(df['SalePrice'].max()),
                            "mean": int(df['SalePrice'].mean()),
                            "median": int(df['SalePrice'].median())
                        },
                        "categorical_features": list(df.select_dtypes(include=['object']).columns),
                        "numerical_features": list(df.select_dtypes(include=['int64', 'float64']).columns),
                        "loaded_from": train_path,
                        "data_source": "actual_file"
                    }
            except Exception as e:
                continue
        
        # If no files found, use verified info from notebooks (fallback)
        return {
            "training_samples": 1460,
            "training_samples_after_preprocessing": 1458,  
            "test_samples": 1459,
            "original_features": 81,
            "final_features": 191,
            "target_column": "SalePrice",
            "price_range": {
                "min": 34900,
                "max": 755000, 
                "mean": 180921,
                "median": 163000
            },
            "categorical_features_count": 43,
            "numerical_features_count": "26-35",
            "data_source": "notebook_verified_fallback"
        }
    
    def _load_model_results(self, base_path: str) -> Dict[str, Any]:
        """Load actual model results from saved files."""
        # Try correct paths based on project structure
        possible_paths = [
            'models/model_summary.json',
            'models/final_model_metadata.json'
        ]
        
        for path_suffix in possible_paths:
            try:
                models_path = os.path.join(base_path, path_suffix)
                if os.path.exists(models_path):
                    with open(models_path, 'r') as f:
                        data = json.load(f)
                        data["loaded_from"] = models_path
                        return data
            except Exception as e:
                continue
        
        return {"error": "Model results not found - may need to run modeling pipeline"}
    
    def _load_feature_descriptions(self, base_path: str) -> str:
        """Load feature descriptions from data_description.txt."""
        # Try correct paths based on project structure
        possible_paths = [
            'docs/data_description.txt',    # From notebooks: ../docs/data_description.txt
            'data/data_description.txt',    # Alternative location
            'data/raw/data_description.txt' # Another possible location
        ]
        
        for path_suffix in possible_paths:
            try:
                desc_path = os.path.join(base_path, path_suffix)
                if os.path.exists(desc_path):
                    with open(desc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        return content[:2000] + "..." if len(content) > 2000 else content  # Truncate if too long
            except Exception as e:
                continue
        return ""
    
    def _load_notebook_summary(self, base_path: str) -> str:
        """Load notebook summary with key insights."""
        # Try to load from project files first  
        possible_paths = [
            'project_summary.md',           # Root level
            'README.md',                    # Common location
            'docs/project_summary.md',      # Docs folder
            'docs/README.md'                # Docs folder
        ]
        
        for path_suffix in possible_paths:
            try:
                summary_path = os.path.join(base_path, path_suffix)
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        return content[:3000] + "..." if len(content) > 3000 else content  # Truncate if too long
            except Exception as e:
                continue
        
        # Fallback to verified summary from notebooks
        return """
VERIFIED NOTEBOOK SUMMARY (from project notebooks):

## Dataset Information
- Original training samples: 1,460
- Training samples after preprocessing: 1,458 (2 outliers removed)
- Test samples: 1,459  
- Original features: 81
- Final engineered features: 191
- Price range: $34,900 - $755,000 (mean: $180,921)

## Model Performance (from 04_modeling notebook)
- Best baseline: CatBoost (RMSE: 0.1166)
- Best individual optimized: CatBoost (RMSE: 0.1143) 
- Final ensemble: Stacking Ensemble (RMSE: 0.1114)
- Validation performance: RMSE $18,713, R² 0.9366, MAPE 8.18%

## Feature Importance (Top 8)
1. GrLivArea (0.0200) - Ground living area
2. OverallQual (0.0125) - Overall quality rating
3. OverallCond (0.0035) - Overall condition
4. TotalFlrSF (0.0034) - Engineered total floor space
5. LotArea (0.0030) - Lot size
6. HouseAge (0.0030) - Engineered house age
7. BsmtFinSF (0.0025) - Engineered basement finished area
8. TotalBaths (0.0017) - Engineered total bathrooms

## Key Insights
- Tree-based models (CatBoost, XGBoost) performed best
- Feature engineering was crucial (4 of top 8 features were engineered)
- Only 2 outliers removed for data quality, preserved market variation
- Ensemble methods provided consistent improvements
- 415 hyperparameter optimization trials conducted
"""
    
    def _load_data_quality_issues(self, base_path: str) -> List[Dict]:
        """Load data quality issues."""
        # Try correct paths based on project structure
        possible_paths = [
            'data/data_quality_issues.csv',
            'data/processed/data_quality_issues.csv'
        ]
        
        for path_suffix in possible_paths:
            try:
                quality_path = os.path.join(base_path, path_suffix)
                if os.path.exists(quality_path):
                    df = pd.read_csv(quality_path)
                    return df.to_dict('records')
            except Exception as e:
                continue
        return []
    
    def _create_system_prompt(self) -> str:
        """Create enhanced system prompt with actual project data."""
        dataset_info = self.project_context.get("dataset_info", {})
        
        prompt = """You are an AI assistant specialized in the Ames Housing Price Prediction project. You have access to the actual project data and results.

IMPORTANT RULES:
- Only use information from the actual project data provided
- Never make up numbers or results
- If you don't have specific information, say so clearly
- Refer to "the project" not "your project"
- Be factual and precise
- Do not use emojis in responses

ACTUAL PROJECT DATA (VERIFIED):
- Original training samples: 1,460 
- Training samples after preprocessing: 1,458 (2 outliers removed for data quality)
- Test samples: 1,459 (unchanged)
- Original features: 81 (training), 80 (test, excluding target)
- Final engineered features: 191 (after feature engineering)
- Target variable: SalePrice (log-transformed for modeling)
- Data types: 43 categorical (object), 26-35 integer, 3-11 float features
- Total combined samples: 2,919 (1,458 + 1,459 after preprocessing)

PREPROCESSING CHANGES (VERIFIED):
- Started with: 1,460 training samples
- Removed: 2 outliers (houses 524 and 1299 - data quality issues, not luxury homes)
- Final training set: 1,458 samples
- Rationale: Only removed clear data errors, kept natural market variation including expensive houses

"""
        
        if dataset_info and "error" not in dataset_info:
            price_range = dataset_info.get('price_range', {})
            if price_range:
                prompt += f"""PRICE STATISTICS (VERIFIED):
- Price range: ${price_range.get('min', 0):,} - ${price_range.get('max', 0):,}
- Mean price: ${price_range.get('mean', 0):,}
- Median price: ${price_range.get('median', 0):,}

"""

        model_results = self.project_context.get("model_results", {})
        if model_results and "error" not in model_results:
            prompt += "MODEL RESULTS: Available from model_summary.json\n"
        else:
            prompt += "MODEL RESULTS: Not currently loaded - refer to notebook files for performance data\n"

        if self.project_context.get("feature_descriptions"):
            prompt += "FEATURE DESCRIPTIONS: Available from data_description.txt\n"

        if self.project_context.get("notebook_summary"):
            prompt += "NOTEBOOK SUMMARY: Available from notebook_summary.md\n"

        prompt += """
PREPROCESSING PIPELINE (VERIFIED):
- Original dataset: 1,460 training + 1,459 test = 2,919 total samples
- Outlier removal: 2 samples removed from training (houses 524 and 1299 for data quality issues)
- Final dataset: 1,458 training + 1,459 test = 2,917 total samples  
- Missing value imputation for 19 features
- Feature type corrections: 3 features (OverallQual, OverallCond, MSSubClass)
- Data consistency validation between train/test sets
- Decision: Only removed clear data errors, preserved natural market variation

RESPONSE GUIDELINES:
1. For dataset questions: Use the verified numbers above (1,460 → 1,458 training samples, etc.)
2. For model performance: Only reference results from loaded files or say "data not available"
3. For feature descriptions: Use data_description.txt content when available
4. For methodology: Reference notebook_summary.md when available
5. Be honest about limitations: If specific data isn't available, say so clearly
6. Use "the project" when referring to the work
7. Never invent performance numbers or model results

Answer questions accurately using only the verified project data shown above."""

        return prompt

    def is_ollama_running(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"].startswith(self.model) for model in models)
            return False
        except requests.RequestException:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model["name"].startswith(self.model):
                        return model
            return {}
        except requests.RequestException:
            return {}
    
    def _enhance_message_with_context(self, message: str) -> str:
        """Enhance user message with relevant project context."""
        message_lower = message.lower()
        enhanced_context = []
        
        # Add dataset info for data questions
        if any(word in message_lower for word in ['data', 'dataset', 'samples', 'features', 'price', 'houses']):
            dataset_info = self.project_context.get("dataset_info", {})
            if dataset_info and "error" not in dataset_info:
                enhanced_context.append(f"DATASET CONTEXT: {json.dumps(dataset_info, indent=2)}")
        
        # Add model results for model-related questions
        if any(word in message_lower for word in ['model', 'performance', 'rmse', 'r2', 'accuracy', 'ensemble']):
            model_results = self.project_context.get("model_results", {})
            if model_results:
                enhanced_context.append(f"MODEL RESULTS CONTEXT: {json.dumps(model_results, indent=2)[:1500]}...")
        
        # Add feature descriptions for specific feature questions
        if any(word in message_lower for word in ['feature', 'describe', 'meaning', 'definition', 'what is']):
            desc = self.project_context.get("feature_descriptions", "")
            if desc:
                enhanced_context.append(f"FEATURE DESCRIPTIONS CONTEXT: {desc[:1000]}...")
        
        # Add notebook summary for methodology questions
        if any(word in message_lower for word in ['how', 'why', 'method', 'approach', 'process', 'notebook']):
            summary = self.project_context.get("notebook_summary", "")
            if summary:
                enhanced_context.append(f"METHODOLOGY CONTEXT: {summary[:1000]}...")
        
        if enhanced_context:
            message += "\n\nAVAILABLE CONTEXT:\n" + "\n\n".join(enhanced_context)
        
        return message
    
    def send_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> Optional[str]:
        """Send a message to Ollama with enhanced project context."""
        if not self.is_ollama_running():
            return None
        
        # Enhance message with project context
        enhanced_message = self._enhance_message_with_context(message)
        
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add chat history
        if chat_history:
            for chat in chat_history[-5:]:  # Keep last 5 exchanges for context
                messages.append({"role": "user", "content": chat["user"]})
                messages.append({"role": "assistant", "content": chat["assistant"]})
        
        # Add current message
        messages.append({"role": "user", "content": enhanced_message})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower temperature for more factual responses
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.RequestException as e:
            return f"Connection error: {str(e)}"

def initialize_chat_session():
    """Initialize chat session in Streamlit."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ollama_chat" not in st.session_state:
        st.session_state.ollama_chat = OllamaChat()

def add_message_to_history(user_message: str, assistant_message: str):
    """Add message pair to chat history."""
    st.session_state.chat_history.append({
        "user": user_message,
        "assistant": assistant_message
    })

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []

def display_chat_history():
    """Display the chat history in the UI."""
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat["user"])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(chat["assistant"])

def get_ollama_status() -> Dict[str, Any]:
    """Get status information about Ollama."""
    chat = OllamaChat()
    
    return {
        "is_running": chat.is_ollama_running(),
        "model": chat.model,
        "base_url": chat.base_url,
        "model_info": chat.get_model_info(),
        "project_data_loaded": bool(chat.project_context.get("dataset_info"))
    }

def get_project_summary() -> Dict[str, Any]:
    """Get summary of loaded project data for display."""
    chat = OllamaChat()
    context = chat.project_context
    
    dataset_info = context.get("dataset_info", {})
    model_results = context.get("model_results", {})
    
    return {
        "dataset_loaded": bool(dataset_info and "error" not in dataset_info),
        "model_results_loaded": bool(model_results and "error" not in model_results),
        "feature_descriptions_loaded": bool(context.get("feature_descriptions")),
        "notebook_summary_loaded": bool(context.get("notebook_summary")),
        "data_quality_loaded": bool(context.get("data_quality_issues")),
        "total_training_samples": dataset_info.get("training_samples", 0) if "error" not in dataset_info else 0,
        "total_features": dataset_info.get("original_features", 0) if "error" not in dataset_info else 0,
        "has_price_data": bool(dataset_info.get("price_range")) if "error" not in dataset_info else False,
        "loaded_from": dataset_info.get("loaded_from", "Unknown") if "error" not in dataset_info else None,
        "dataset_error": dataset_info.get("error") if "error" in dataset_info else None,
        "model_error": model_results.get("error") if "error" in model_results else None
    }