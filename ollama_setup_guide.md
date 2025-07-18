# Ollama Setup Guide - AI Chat Integration

This guide shows you how to install and configure Ollama to work with your Streamlit AI chat.

## Step 1: Install Ollama

### macOS

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows

1. Go to https://ollama.ai/download
2. Download Ollama for Windows
3. Run the installation file and follow the instructions

### Linux

```bash
# Download and install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

## Step 2: Start Ollama Service

### macOS and Linux

```bash
# Start Ollama service
ollama serve
```

### Windows

Ollama starts automatically after installation. If not, you can start it from the Start menu.

## Step 3: Download Llama3.2 Model

Open a new terminal/command prompt and run:

```bash
# Download llama3.2 model (recommended model for the project)
ollama pull llama3.2
```

This will download approximately 2GB of data, so make sure you have a stable internet connection.

## Step 4: Test Ollama Installation

Test that everything works by running:

```bash
# Test that the model works
ollama run llama3.2 "Hello, can you say hello back?"
```

You should get a response from the AI in English.

## Step 5: Start Your Streamlit App

Now you can start your Streamlit app:

```bash
# From the HousePrediction folder
cd streamlit
streamlit run app.py
```

Go to the "AI Chat Assistant" page and check that the status shows "✅ Ollama is running".

## Troubleshooting

### Problem: "Ollama is not running"

**Solution:**

1. Check that Ollama service is running:

   ```bash
   # macOS/Linux
   ollama serve

   # Windows: Start Ollama from Start menu
   ```

2. Verify that llama3.2 model is installed:
   ```bash
   ollama list
   ```

### Problem: "Connection error"

**Solution:**

1. Check that Ollama is running on the default port (11434):

   ```bash
   # Test direct API access
   curl http://localhost:11434/api/tags
   ```

2. If you're using a different port, update `base_url` in `ai_chat_utils.py`

### Problem: Model not found

**Solution:**

```bash
# Delete and re-download model
ollama rm llama3.2
ollama pull llama3.2
```

### Problem: Slow response

**Solution:**

- Llama3.2 model is optimized for balance between speed and quality
- First response can be slow while model loads
- Subsequent responses should be faster

## Advanced Settings

### Change Model (optional)

If you want to try other models:

```bash
# Smaller model (faster, but less capable)
ollama pull llama3.2:1b

# Larger model (more capable, but slower)
ollama pull llama3.1:8b
```

Then update the `model` parameter in `ai_chat_utils.py`:

```python
def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
```

### Change System Prompt

You can customize the AI's behavior by changing `system_prompt` in `ai_chat_utils.py`.

### Memory Usage

- Llama3.2 uses around 2-4GB RAM
- If you have less than 8GB RAM, consider a smaller model like `llama3.2:1b`

## Test Questions

Try these questions to test the AI:

1. "What are the most important features for house price prediction?"
2. "Explain what ensemble methods are"
3. "How do you handle missing values in machine learning?"
4. "What's the difference between overfitting and underfitting?"

## Support

If you encounter problems:

1. Check Ollama documentation: https://ollama.ai/docs
2. Verify that all dependencies are installed
3. Restart Ollama service
4. Try re-downloading the model

---

**Congratulations!** Your AI chat assistant is now ready to help with your house price prediction project! 🤖✨
