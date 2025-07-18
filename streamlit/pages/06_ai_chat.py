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
    get_ollama_status
)

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    layout="wide"
)

st.title("AI Chat Assistant")
st.markdown("---")

st.markdown("""
Ask the AI questions about the house price prediction project, data science, or machine learning in general.
The AI has knowledge about your project and can help explain the analysis and methods.
""")

# Initialize chat session
initialize_chat_session()

# Check Ollama status
status = get_ollama_status()

# Sidebar with status and controls
with st.sidebar:
    st.header("AI Chat Status")
    
    if status["is_running"]:
        st.success("✅ Ollama is running")
        st.info(f"**Model:** {status['model']}")
        
        # Model info if available
        if status["model_info"]:
            model_info = status["model_info"]
            st.write(f"**Size:** {model_info.get('size', 'Unknown')}")
            if 'modified_at' in model_info:
                modified_date = datetime.fromisoformat(model_info['modified_at'].replace('Z', '+00:00'))
                st.write(f"**Modified:** {modified_date.strftime('%Y-%m-%d')}")
    else:
        st.error("❌ Ollama is not running")
        st.warning("Start Ollama first and make sure llama3.2 is installed.")
    
    st.markdown("---")
    
    # Chat controls
    st.header("Chat Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()
    
    # Chat statistics
    chat_count = len(st.session_state.chat_history)
    st.metric("Message Count", chat_count)
    
    st.markdown("---")
    
    # Project context
    st.header("Project Context")
    st.markdown("""
    **House Price Prediction Project:**
    - Ames Housing dataset
    - 1,460 training samples
    - 81 features
    - Data preprocessing
    - Feature engineering
    - Machine learning models
    - Ensemble methods
    """)

# Main chat interface
if not status["is_running"]:
    st.error("""
    **Ollama is not available!**
    
    Follow these steps:
    1. Install Ollama from https://ollama.ai
    2. Start Ollama service
    3. Download llama3.2 model: `ollama pull llama3.2`
    4. Reload this page
    """)
else:
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = st.session_state.ollama_chat.send_message(
                    user_input, 
                    st.session_state.chat_history
                )
                
                if response:
                    st.write(response)
                    
                    # Add to history
                    add_message_to_history(user_input, response)
                else:
                    error_msg = "Sorry, I couldn't process your question. Please check that Ollama is running correctly."
                    st.error(error_msg)
                    add_message_to_history(user_input, error_msg)

# Example questions
st.markdown("---")
st.header("Example Questions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**About the Project:**")
    example_questions_1 = [
        "What are the most important features for house price prediction?",
        "How did you handle missing values in the dataset?",
        "Which models performed best and why?",
        "What are ensemble methods and why did you use them?"
    ]
    
    for question in example_questions_1:
        if st.button(f"💬 {question}", key=f"q1_{hash(question)}"):
            st.session_state.example_question = question
            st.rerun()

with col2:
    st.markdown("**General Data Science:**")
    example_questions_2 = [
        "What's the difference between overfitting and underfitting?",
        "How do you evaluate a regression model?",
        "What is cross-validation and why is it important?",
        "How do you handle outliers in data?"
    ]
    
    for question in example_questions_2:
        if st.button(f"💬 {question}", key=f"q2_{hash(question)}"):
            st.session_state.example_question = question
            st.rerun()

# Handle example question clicks
if hasattr(st.session_state, 'example_question') and status["is_running"]:
    example_q = st.session_state.example_question
    
    # Display user message
    with st.chat_message("user"):
        st.write(example_q)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            response = st.session_state.ollama_chat.send_message(
                example_q, 
                st.session_state.chat_history
            )
            
            if response:
                st.write(response)
                add_message_to_history(example_q, response)
            else:
                error_msg = "Sorry, I couldn't process your question."
                st.error(error_msg)
                add_message_to_history(example_q, error_msg)
    
    # Clear the example question
    del st.session_state.example_question

# Footer
st.markdown("---")
st.markdown("""
**AI Chat Assistant** | Powered by Ollama & Llama3.2

*The AI has knowledge about your house price prediction project and can help with questions about data science and machine learning.*
""")