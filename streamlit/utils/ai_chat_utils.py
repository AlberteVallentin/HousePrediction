import requests
import json
import streamlit as st
import pandas as pd
import os
from typing import List, Dict, Any, Optional

class OllamaChat:
    """Simple Ollama chat integration for house price prediction project."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        self.system_prompt = """You are an AI assistant helping with a house price prediction project using the Ames Housing dataset. 

You have access to the actual project files and can reference specific information when users ask questions:

**PROJECT STRUCTURE:**
- Data files: /Users/albertevallentin/Developer/HousePrediction/data/
  - train.csv (training data)
  - test.csv (test data)
  - data_description.txt (feature descriptions)
- Notebooks: /Users/albertevallentin/Developer/HousePrediction/notebooks/
  - 01_data_exploration.ipynb
  - 02_data_preprocessing.ipynb
  - 03_feature_engineering.ipynb
  - 04_modeling.ipynb
- Models: /Users/albertevallentin/Developer/HousePrediction/models/
  - Various model files and metadata
- Streamlit app: /Users/albertevallentin/Developer/HousePrediction/streamlit/

**HOW TO HELP USERS:**

1. **For data questions**: Explain that you can't directly read CSV files, but you know the project has:
   - 1,460 training samples with 79 original features
   - Feature engineering created 191 final features
   - Key features include GrLivArea, OverallQual, OverallCond, etc.

2. **For feature descriptions**: Refer users to the data_description.txt file for detailed feature explanations

3. **For specific analysis**: Reference the relevant notebook:
   - Data exploration → 01_data_exploration.ipynb
   - Data preprocessing → 02_data_preprocessing.ipynb  
   - Feature engineering → 03_feature_engineering.ipynb
   - Model results → 04_modeling.ipynb

4. **For model results and feature importance**: I can access the actual results from your project files:
   - Model performance data from models/model_summary.json
   - Feature importance from your notebooks
   - When you ask about models, I'll automatically load and reference your actual results
   - When you ask about features, I'll provide information from your data_description.txt

5. **How to get accurate information**: 
   - Ask about specific models and I'll read your actual results
   - Ask about features and I'll reference your data descriptions
   - Ask about data and I'll analyze your actual dataset
   - All information comes from your project files, not assumptions

**IMPORTANT GUIDANCE:**
- When users ask about specific data or analysis, guide them to the relevant files
- If they want to see actual data, suggest they check the CSV files or notebooks
- For detailed feature explanations, refer to data_description.txt
- For methodology, refer to the specific notebooks
- Always be accurate about what information you have access to

Answer in English and be helpful and informative."""
    
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
    
    def send_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> Optional[str]:
        """Send a message to Ollama and get response."""
        if not self.is_ollama_running():
            return None
        
        # Enhance message with project context if relevant
        enhanced_message = enhance_message_with_context(message, chat_history)
        
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add chat history
        if chat_history:
            for chat in chat_history:
                messages.append({"role": "user", "content": chat["user"]})
                messages.append({"role": "assistant", "content": chat["assistant"]})
        
        # Add current message (enhanced with context)
        messages.append({"role": "user", "content": enhanced_message})
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.RequestException as e:
            return f"Connection error: {str(e)}"
    
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
        "model_info": chat.get_model_info()
    }

def get_project_data_info() -> Dict[str, Any]:
    """Get information about project data files."""
    base_path = os.path.join(os.path.dirname(__file__), '../../')
    
    info = {
        "data_files": [],
        "notebooks": [],
        "models": []
    }
    
    # Check data files
    data_path = os.path.join(base_path, 'data')
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.endswith('.csv') or file.endswith('.txt'):
                info["data_files"].append(file)
    
    # Check notebooks
    notebooks_path = os.path.join(base_path, 'notebooks')
    if os.path.exists(notebooks_path):
        for file in os.listdir(notebooks_path):
            if file.endswith('.ipynb'):
                info["notebooks"].append(file)
    
    # Check models
    models_path = os.path.join(base_path, 'models')
    if os.path.exists(models_path):
        for file in os.listdir(models_path):
            if file.endswith('.json') or file.endswith('.joblib'):
                info["models"].append(file)
    
    return info

def get_data_description() -> str:
    """Get data description from data_description.txt if available."""
    try:
        data_desc_path = os.path.join(os.path.dirname(__file__), '../../data/data_description.txt')
        if os.path.exists(data_desc_path):
            with open(data_desc_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "Data description file not found."
    except Exception as e:
        return f"Error reading data description: {str(e)}"

def get_dataset_summary() -> Dict[str, Any]:
    """Get basic summary of the training dataset."""
    try:
        train_path = os.path.join(os.path.dirname(__file__), '../../data/train.csv')
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "target_column": "SalePrice" if "SalePrice" in df.columns else None,
                "missing_values": df.isnull().sum().sum(),
                "sample_data": df.head().to_dict('records') if len(df) > 0 else []
            }
        return {"error": "Training data file not found"}
    except Exception as e:
        return {"error": f"Error reading training data: {str(e)}"}

def get_model_results() -> Dict[str, Any]:
    """Get model results from saved files."""
    try:
        results = {}
        
        # Try to read model summary
        models_path = os.path.join(os.path.dirname(__file__), '../../models/')
        summary_path = os.path.join(models_path, 'model_summary.json')
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                results['model_summary'] = json.load(f)
        
        # Try to read feature importance/order
        feature_order_path = os.path.join(models_path, 'feature_order.json')
        if os.path.exists(feature_order_path):
            with open(feature_order_path, 'r') as f:
                results['feature_info'] = json.load(f)
        
        # List available model files
        if os.path.exists(models_path):
            model_files = [f for f in os.listdir(models_path) if f.endswith('.json') or f.endswith('.joblib')]
            results['available_files'] = model_files
        
        return results
    except Exception as e:
        return {"error": f"Error reading model results: {str(e)}"}

def enhance_message_with_context(message: str, chat_history: List[Dict[str, str]] = None) -> str:
    """Enhance user message with relevant project context if needed."""
    message_lower = message.lower()
    
    # If user asks about data, add dataset summary
    if any(keyword in message_lower for keyword in ['data', 'dataset', 'features', 'columns']):
        dataset_summary = get_dataset_summary()
        if 'error' not in dataset_summary:
            context = f"\n\nCONTEXT: The training dataset has {dataset_summary['shape'][0]} rows and {dataset_summary['shape'][1]} columns. Key columns include: {', '.join(dataset_summary['columns'][:10])}..."
            message += context
    
    # If user asks about models, performance, or results, add model results
    if any(keyword in message_lower for keyword in ['model', 'performance', 'rmse', 'results', 'best', 'accuracy', 'ensemble', 'random forest', 'catboost']):
        model_results = get_model_results()
        if 'error' not in model_results and 'model_summary' in model_results:
            context = f"\n\nCONTEXT: Available model results: {json.dumps(model_results['model_summary'], indent=2)[:2000]}..."
            message += context
    
    # If user asks about feature descriptions, add data description
    if any(keyword in message_lower for keyword in ['feature description', 'what is', 'describe']):
        if 'feature' in message_lower or any(col in message_lower for col in ['overall', 'garage', 'basement', 'kitchen']):
            data_desc = get_data_description()[:1000]  # Limit to first 1000 chars
            if "Data description file not found" not in data_desc:
                context = f"\n\nCONTEXT: Here's part of the data description: {data_desc}..."
                message += context
    
    return message