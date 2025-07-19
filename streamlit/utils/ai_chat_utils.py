import requests
import json
import streamlit as st
import pandas as pd
import os
import numpy as np
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
            "data_quality_issues": [],
            "actual_data_samples": {},
            "preprocessing_summary": {},
            "ollama_context": ""  # New: dedicated context file
        }
        
        # Base path - adjust based on where the utils are located
        base_path = os.path.join(os.path.dirname(__file__), '../..')
        
        # Load dedicated Ollama context file (PRIORITY)
        context["ollama_context"] = self._load_ollama_context(base_path)
        
        # Load actual dataset information
        context["dataset_info"] = self._load_dataset_info(base_path)
        context["actual_data_samples"] = self._load_data_samples(base_path)
        
        # Load model results if available
        context["model_results"] = self._load_model_results(base_path)
        
        # Load feature descriptions
        context["feature_descriptions"] = self._load_feature_descriptions(base_path)
        
        # Load notebook summary
        context["notebook_summary"] = self._load_notebook_summary(base_path)
        
        # Load data quality issues
        context["data_quality_issues"] = self._load_data_quality_issues(base_path)
        
        return context
    
    def _load_ollama_context(self, base_path: str) -> str:
        """Load dedicated Ollama context file with comprehensive project information."""
        # Try to find ollama_context.md in project root
        possible_paths = [
            '../../ollama_context.md',          # From streamlit/utils/
            '../ollama_context.md',             # From streamlit/
            'ollama_context.md',                # From project root
            '../../docs/ollama_context.md',     # Alternative in docs
            '../docs/ollama_context.md',        # Alternative in docs
            'docs/ollama_context.md'            # Alternative in docs
        ]
        
        for path_suffix in possible_paths:
            try:
                context_path = os.path.join(base_path, path_suffix)
                if os.path.exists(context_path):
                    with open(context_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        return content  # Return full content for Ollama
            except Exception as e:
                continue
        
        # Return empty string if file not found
        return ""
    
    def _load_dataset_info(self, base_path: str) -> Dict[str, Any]:
        """Load actual dataset information from CSV files."""
        try:
            info = {"source": "actual_files", "loaded_successfully": True}
            
            # Try project structure paths based on actual filestructure.md
            # From streamlit/utils/, we need to go up to project root
            possible_locations = [
                "../../data/raw/",           # From streamlit/utils/ to project root
                "../data/raw/",              # From streamlit/ to project root  
                "data/raw/",                 # If running from project root
                "../../data/processed/",     # Processed files
                "../data/processed/",        # Processed files
                "data/processed/"            # Processed files
            ]
            
            data_paths = []
            
            for location in possible_locations:
                train_path = os.path.join(base_path, location, "train.csv")
                test_path = os.path.join(base_path, location, "test.csv")
                if os.path.exists(train_path) and os.path.exists(test_path):
                    data_paths = [train_path, test_path]
                    info["data_location"] = location
                    break
            
            if not data_paths:
                return {"error": "Dataset files not found", "searched_locations": possible_locations}
            
            # Load the actual data
            if len(data_paths) >= 2:
                df_train = pd.read_csv(data_paths[0])
                df_test = pd.read_csv(data_paths[1])
                
                # Calculate actual statistics
                info.update({
                    "original_training_samples": len(df_train),
                    "original_test_samples": len(df_test),
                    "original_training_features": df_train.shape[1],
                    "original_test_features": df_test.shape[1],
                    "has_target": "SalePrice" in df_train.columns,
                    "loaded_from": data_paths
                })
                
                if "SalePrice" in df_train.columns:
                    price_stats = {
                        "min": int(df_train["SalePrice"].min()),
                        "max": int(df_train["SalePrice"].max()),
                        "mean": int(df_train["SalePrice"].mean()),
                        "median": int(df_train["SalePrice"].median()),
                        "std": int(df_train["SalePrice"].std())
                    }
                    info["price_range"] = price_stats
                
                # Check for processed files based on actual structure
                processed_files = {
                    "train_cleaned.csv": "../../data/processed/train_cleaned.csv",
                    "test_cleaned.csv": "../../data/processed/test_cleaned.csv", 
                    "X_train_final.csv": "../../data/processed/X_train_final.csv",
                    "X_test_final.csv": "../../data/processed/X_test_final.csv",
                    "y_train_final.csv": "../../data/processed/y_train_final.csv"
                }
                
                processed_info = {}
                
                for filename, relative_path in processed_files.items():
                    proc_path = os.path.join(base_path, relative_path)
                    if os.path.exists(proc_path):
                        try:
                            df_proc = pd.read_csv(proc_path)
                            processed_info[filename] = {
                                "samples": len(df_proc),
                                "features": df_proc.shape[1],
                                "path": relative_path
                            }
                        except Exception as e:
                            processed_info[filename] = {"error": str(e)}
                
                if processed_info:
                    info["processed_files"] = processed_info
                
                return info
            
        except Exception as e:
            return {"error": f"Error loading dataset: {str(e)}"}
        
        return {"error": "Could not load dataset information"}
    
    def _load_data_samples(self, base_path: str) -> Dict[str, Any]:
        """Load sample data for context."""
        try:
            # Try project structure paths
            possible_locations = [
                "../../data/raw/train.csv",     # From streamlit/utils/
                "../data/raw/train.csv",        # From streamlit/
                "data/raw/train.csv"            # From project root
            ]
            
            train_path = None
            for location in possible_locations:
                full_path = os.path.join(base_path, location)
                if os.path.exists(full_path):
                    train_path = full_path
                    break
            
            if train_path:
                df_sample = pd.read_csv(train_path).head(3)  # Just first 3 rows
                
                # Convert to dict and clean for JSON serialization
                sample_dict = {}
                for col in df_sample.columns[:10]:  # First 10 columns only
                    sample_dict[col] = df_sample[col].astype(str).tolist()
                
                return {
                    "sample_data": sample_dict,
                    "sample_size": len(df_sample),
                    "total_columns_shown": min(10, len(df_sample.columns))
                }
        except Exception as e:
            pass
        
        return {}
    
    def _load_model_results(self, base_path: str) -> Dict[str, Any]:
        """Load model results from saved files."""
        # Try project structure paths based on actual filestructure.md
        possible_paths = [
            "../../models/model_summary.json",           # From streamlit/utils/
            "../models/model_summary.json",              # From streamlit/
            "models/model_summary.json",                 # From project root
            "../../models/final_model_metadata.json",    # Alternative file
            "../models/final_model_metadata.json",       # Alternative file
            "models/final_model_metadata.json"           # Alternative file
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
        
        return {"error": "Model results not found - check if modeling notebook has been run"}
    
    def _load_feature_descriptions(self, base_path: str) -> str:
        """Load feature descriptions from data_description.txt."""
        # Based on actual project structure: docs/data_description.txt
        possible_paths = [
            '../../docs/data_description.txt',      # From streamlit/utils/
            '../docs/data_description.txt',         # From streamlit/
            'docs/data_description.txt',            # From project root
            '../../data/data_description.txt',      # Alternative location
            '../data/data_description.txt',         # Alternative location
            'data/data_description.txt'             # Alternative location
        ]
        
        for path_suffix in possible_paths:
            try:
                desc_path = os.path.join(base_path, path_suffix)
                if os.path.exists(desc_path):
                    with open(desc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Return first 3000 chars to avoid context limits
                        return content[:3000] + "..." if len(content) > 3000 else content
            except Exception as e:
                continue
        return ""
    
    def _load_notebook_summary(self, base_path: str) -> str:
        """Load notebook summary with key insights."""
        # Based on actual project structure: docs/notebook_summary.md
        possible_paths = [
            '../../docs/notebook_summary.md',       # From streamlit/utils/
            '../docs/notebook_summary.md',          # From streamlit/
            'docs/notebook_summary.md',             # From project root
            '../../project_summary.md',             # Alternative at root
            '../project_summary.md',                # Alternative at root  
            'project_summary.md',                   # Alternative at root
            '../../README.md',                      # Fallback
            '../README.md',                         # Fallback
            'README.md'                             # Fallback
        ]
        
        for path_suffix in possible_paths:
            try:
                summary_path = os.path.join(base_path, path_suffix)
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Return first 4000 chars to avoid context limits
                        return content[:4000] + "..." if len(content) > 4000 else content
            except Exception as e:
                continue
        return ""
    
    def _load_data_quality_issues(self, base_path: str) -> List[Dict]:
        """Load data quality issues from CSV."""
        possible_paths = [
            'data/data_quality_issues.csv',
            '../data/data_quality_issues.csv',
            'data_quality_issues.csv'
        ]
        
        for path_suffix in possible_paths:
            try:
                issues_path = os.path.join(base_path, path_suffix)
                if os.path.exists(issues_path):
                    df_issues = pd.read_csv(issues_path)
                    return df_issues.to_dict('records')[:10]  # First 10 issues only
            except Exception as e:
                continue
        return []
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt with actual project data."""
        ollama_context = self.project_context.get("ollama_context", "")
        
        # If we have the dedicated context file, use it as primary source
        if ollama_context:
            prompt = f"""You are an AI assistant specialized in the Ames Housing Price Prediction project. You have access to comprehensive, verified project information.

CRITICAL INSTRUCTIONS:
- Use ONLY the verified information provided in the project context below
- Never make up performance numbers, statistics, or results
- If specific information isn't in the context, explicitly state this
- Be precise and factual with exact numbers from the context
- Refer to "the project" when discussing the work
- Don't use emojis in responses
- When citing performance or statistics, use the exact figures provided

COMPREHENSIVE PROJECT CONTEXT:
{ollama_context}

RESPONSE GUIDELINES:
1. Always use the verified data from the context above
2. For any specific numbers or results, reference the exact figures provided
3. If asked about something not covered in the context, clearly state "This information is not available in the project documentation"
4. Focus on providing insights based on the actual project results
5. Connect individual facts to the broader project objectives and learnings

You are an expert on this specific project and should answer confidently based on the comprehensive context provided above."""

        else:
            # Fallback to dynamic loading if context file not available
            dataset_info = self.project_context.get("dataset_info", {})
            model_results = self.project_context.get("model_results", {})
            
            prompt = """You are an AI assistant specialized in the Ames Housing Price Prediction project. You have access to actual project data and results.

CRITICAL INSTRUCTIONS:
- Only use information from the verified project data provided below
- Never make up performance numbers, statistics, or results
- If specific information isn't available, explicitly state this
- Be precise and factual
- Refer to "the project" when discussing the work
- Don't use emojis in responses

"""

            # Add dataset information if available
            if dataset_info and "error" not in dataset_info:
                prompt += f"""VERIFIED DATASET INFORMATION:
- Original training samples: {dataset_info.get('original_training_samples', 'Unknown')}
- Original test samples: {dataset_info.get('original_test_samples', 'Unknown')}
- Training features: {dataset_info.get('original_training_features', 'Unknown')}
- Test features: {dataset_info.get('original_test_features', 'Unknown')}
- Target variable: {'SalePrice (available)' if dataset_info.get('has_target') else 'SalePrice (not in test set)'}

"""
                
                # Add price statistics if available
                price_range = dataset_info.get('price_range', {})
                if price_range:
                    prompt += f"""ACTUAL PRICE STATISTICS:
- Price range: ${price_range.get('min', 0):,} to ${price_range.get('max', 0):,}
- Mean price: ${price_range.get('mean', 0):,}
- Median price: ${price_range.get('median', 0):,}

"""
            
            # Add model results if available
            if model_results and "error" not in model_results:
                final_model = model_results.get('final_model', {})
                if final_model:
                    prompt += f"""VERIFIED MODEL RESULTS:
- Final model: {final_model.get('final_model_name', 'Unknown')}
- Final CV RMSE: {final_model.get('final_cv_rmse', 'Unknown')}

"""
            
            prompt += """
RESPONSE GUIDELINES:
1. Use only the verified data shown above
2. If information isn't available, state this clearly
3. Focus on factual information from actual project files
4. Be precise with numbers and avoid approximations"""

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
        enhanced_message = f"User question: {message}\n\n"
        
        # Add relevant context based on message content
        if any(keyword in message.lower() for keyword in ['dataset', 'data', 'samples', 'features']):
            dataset_info = self.project_context.get("dataset_info", {})
            if dataset_info and "error" not in dataset_info:
                enhanced_message += f"Dataset context: {json.dumps(dataset_info, indent=2)}\n\n"
        
        if any(keyword in message.lower() for keyword in ['model', 'performance', 'rmse', 'accuracy']):
            model_results = self.project_context.get("model_results", {})
            if model_results and "error" not in model_results:
                # Add key model results
                final_model = model_results.get('final_model', {})
                perf_comparison = model_results.get('performance_comparison', {})
                
                enhanced_message += "Model performance context:\n"
                if final_model:
                    enhanced_message += f"Final model: {final_model}\n"
                if perf_comparison:
                    enhanced_message += f"Performance comparison: {dict(list(perf_comparison.items())[:3])}\n"
                enhanced_message += "\n"
        
        if any(keyword in message.lower() for keyword in ['feature', 'variable', 'description']):
            feature_desc = self.project_context.get("feature_descriptions", "")
            if feature_desc:
                enhanced_message += f"Feature descriptions available: {len(feature_desc)} characters loaded\n\n"
        
        return enhanced_message
    
    def send_message(self, message: str, conversation_history: List[Dict] = None) -> Optional[str]:
        """Send message to Ollama and get response."""
        if not self.is_ollama_running():
            return None
        
        try:
            # Enhance message with project context
            enhanced_message = self._enhance_message_with_context(message)
            
            # Prepare conversation for Ollama
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            if conversation_history:
                for entry in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({"role": "user", "content": entry["user"]})
                    messages.append({"role": "assistant", "content": entry["assistant"]})
            
            # Add current message
            messages.append({"role": "user", "content": enhanced_message})
            
            # Send to Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.RequestException as e:
            return f"Connection error: {str(e)}"

# Streamlit integration functions
def initialize_chat_session():
    """Initialize chat session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "ollama_chat" not in st.session_state:
        st.session_state.ollama_chat = OllamaChat()

def add_message_to_history(user_message: str, assistant_response: str):
    """Add message to chat history."""
    st.session_state.chat_history.append({
        "user": user_message,
        "assistant": assistant_response,
        "timestamp": pd.Timestamp.now()
    })

def clear_chat_history():
    """Clear chat history."""
    st.session_state.chat_history = []

def display_chat_history():
    """Display chat history in sidebar."""
    if st.session_state.chat_history:
        st.markdown("### Recent Conversations")
        
        # Show last few conversations
        for i, entry in enumerate(st.session_state.chat_history[-3:]):
            with st.expander(f"Q: {entry['user'][:50]}...", expanded=False):
                st.markdown(f"**You:** {entry['user']}")
                st.markdown(f"**AI:** {entry['assistant'][:200]}...")

def get_ollama_status() -> Dict[str, Any]:
    """Get Ollama status and model information."""
    chat = OllamaChat()
    
    return {
        "is_running": chat.is_ollama_running(),
        "model": chat.model,
        "base_url": chat.base_url,
        "model_info": chat.get_model_info(),
        "project_data_loaded": bool(chat.project_context.get("dataset_info")) and "error" not in chat.project_context.get("dataset_info", {})
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
        "total_training_samples": dataset_info.get("original_training_samples", 0) if "error" not in dataset_info else 0,
        "total_test_samples": dataset_info.get("original_test_samples", 0) if "error" not in dataset_info else 0,
        "total_features": dataset_info.get("original_training_features", 0) if "error" not in dataset_info else 0,
        "has_price_data": bool(dataset_info.get("price_range")) if "error" not in dataset_info else False,
        "loaded_from": dataset_info.get("loaded_from", "Unknown") if "error" not in dataset_info else None,
        "dataset_error": dataset_info.get("error") if "error" in dataset_info else None,
        "model_error": model_results.get("error") if "error" in model_results else None,
        "processed_files": dataset_info.get("processed_files", {}) if "error" not in dataset_info else {}
    }