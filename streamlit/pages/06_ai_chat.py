import streamlit as st
import sys
import os
from datetime import datetime

# Add the parent directory to access utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.ai_chat_utils import (
    initialize_chat_session,
    add_message_to_history,
    clear_chat_history,
    display_chat_history,
    get_ollama_status,
    get_project_summary
)

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant - House Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-good {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #4caf50;
    }
    .status-bad {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #f44336;
    }
    .project-stats {
        background-color: #f0f4ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3f51b5;
    }
    .question-button {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.3rem;
        cursor: pointer;
        width: 100%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Intelligent assistant with access to the house price prediction project data and results**")

# Initialize chat session
initialize_chat_session()

# Check Ollama status and project data
status = get_ollama_status()
project_summary = get_project_summary()

# Sidebar with status and project info
with st.sidebar:
    st.markdown("## System Status")
    
    # Ollama status
    if status["is_running"]:
        st.markdown('<div class="status-good">Ollama is running</div>', unsafe_allow_html=True)
        st.info(f"**Model:** {status['model']}")
        
        # Model info if available
        if status["model_info"]:
            model_info = status["model_info"]
            if 'size' in model_info:
                size_gb = round(model_info['size'] / (1024**3), 1)
                st.write(f"**Size:** {size_gb} GB")
    else:
        st.markdown('<div class="status-bad">Ollama not running</div>', unsafe_allow_html=True)
        st.error("Start Ollama and install llama3.2 model")
    
    st.markdown("---")
    
    # Project data status
    st.markdown("## Project Data Status")
    
    data_status = "Available" if project_summary["dataset_loaded"] else "Not found"
    st.write(f"**Dataset:** {data_status}")
    if project_summary["total_training_samples"]:
        st.write(f"- {project_summary['total_training_samples']} training samples")
        st.write(f"- {project_summary['total_features']} original features")
    
    model_status = "Available" if project_summary["model_results_loaded"] else "Not found"
    st.write(f"**Model Results:** {model_status}")
    
    desc_status = "Available" if project_summary["feature_descriptions_loaded"] else "Not found"
    st.write(f"**Feature Descriptions:** {desc_status}")
    
    summary_status = "Available" if project_summary["notebook_summary_loaded"] else "Not found"
    st.write(f"**Notebook Summary:** {summary_status}")
    
    st.markdown("---")
    
    # Chat controls
    st.markdown("## Chat Controls")
    
    if st.button("Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()
    
    # Chat statistics
    chat_count = len(st.session_state.chat_history)
    st.metric("Messages", chat_count)

# Main content area
if not status["is_running"]:
    st.error("""
    **Ollama is not available**
    
    To use the AI assistant, please:
    1. Install Ollama from https://ollama.ai
    2. Start Ollama service in terminal
    3. Download the model: `ollama pull llama3.2`
    4. Refresh this page
    
    The AI assistant uses Ollama to run the language model locally.
    """)
    
    # Show project data status even without Ollama
    if project_summary["dataset_loaded"]:
        st.info("The project data is loaded and ready for AI analysis once Ollama is running.")

else:
    # Project data summary
    if project_summary["dataset_loaded"]:
        st.markdown("""
        <div class="project-stats">
        <h4>Project Data Loaded</h4>
        <p>The AI has access to the actual project data and can answer specific questions about:</p>
        <ul>
            <li>Dataset characteristics and statistics</li>
            <li>Model performance and comparisons</li>
            <li>Feature descriptions and importance</li>
            <li>Methodology and preprocessing steps</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Project data not fully loaded. The AI will have limited knowledge about the specific project.")
        
        # Show debug info for troubleshooting
        if project_summary.get("dataset_error"):
            st.error(f"Dataset loading error: {project_summary['dataset_error']}")
        if project_summary.get("model_error"):
            st.error(f"Model results loading error: {project_summary['model_error']}")
            
        st.info("""
        **Troubleshooting:** The AI is looking for project files in these locations:
        - train.csv (uploaded files or data/raw/ folder)
        - model_summary.json (models/ folder)
        - data_description.txt (data/ folder)
        - notebook_summary.md (root folder)
        
        Make sure these files are accessible from the Streamlit app directory.
        """)
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Ask about the house price prediction project...")
    
    # Handle example questions from buttons
    if 'example_question' in st.session_state:
        user_input = st.session_state.example_question
        del st.session_state.example_question
    
    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing project data..."):
                response = st.session_state.ollama_chat.send_message(
                    user_input, 
                    st.session_state.chat_history
                )
                
                if response:
                    st.write(response)
                    add_message_to_history(user_input, response)
                else:
                    error_msg = "Sorry, I couldn't process the question. Please check that Ollama is running correctly."
                    st.error(error_msg)
                    add_message_to_history(user_input, error_msg)

# Example questions section
st.markdown("---")
st.markdown("## Example Questions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**About Project Data:**")
    
    project_questions = [
        "What is the size and structure of the dataset?",
        "What was the price range of houses in the dataset?",
        "How many features were in the original dataset?",
        "What preprocessing steps were applied to the data?",
        "Which features were most important for predictions?",
        "How were missing values handled in the project?"
    ]
    
    for question in project_questions:
        if st.button(question, key=f"proj_{hash(question)}", use_container_width=True):
            st.session_state.example_question = question
            st.rerun()

with col2:
    st.markdown("**About Models and Methods:**")
    
    ml_questions = [
        "Which models were tested in the project?",
        "How did different models perform?",
        "What evaluation metrics were used?",
        "How was model validation performed?",
        "What ensemble methods were implemented?",
        "How were hyperparameters optimized?"
    ]
    
    for question in ml_questions:
        if st.button(question, key=f"ml_{hash(question)}", use_container_width=True):
            st.session_state.example_question = question
            st.rerun()

# Additional specific questions
st.markdown("**Methodology and Insights:**")

methodology_questions = [
    "What feature engineering was performed?",
    "How were outliers identified and handled?",
    "What insights were gained about house price drivers?",
    "How does the final model performance compare to baselines?",
    "What were the key learnings from the project?",
    "How was the model validated for reliability?"
]

col1, col2, col3 = st.columns(3)
for i, question in enumerate(methodology_questions):
    col = [col1, col2, col3][i % 3]
    with col:
        if st.button(question, key=f"method_{hash(question)}", use_container_width=True):
            st.session_state.example_question = question
            st.rerun()

# Footer information
st.markdown("---")
st.markdown("""
### Tips for Better Conversations

- **Be specific:** Ask about particular aspects of the project
- **Request comparisons:** Compare different models or approaches used
- **Ask for explanations:** Understand the reasoning behind decisions
- **Explore results:** Dive into performance metrics and insights

### What the AI Knows

The AI has access to project files including dataset information, model results, 
feature descriptions, and methodology documentation. It will only provide information 
based on actual project data and will indicate when specific information is not available.

---
*AI Assistant powered by Ollama & Llama 3.2 | Using actual project data*
""")

# Debug info (only show if there are issues)
if st.checkbox("Show Debug Info", value=False):
    st.markdown("### Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Ollama Status:**")
        st.json(status)
    
    with col2:
        st.write("**Project Summary:**")
        st.json(project_summary)