# Chatbot with Memory

This project is a chatbot application that uses a language model and memory to provide context-aware responses. The chatbot is built using the `langchain` library and integrates with Google Vertex AI for embeddings. The conversation context is stored in a vector store using Chroma. The application also includes a Streamlit web interface for interacting with the chatbot.

## Features

- Context-aware chatbot using memory
- Integration with Google Vertex AI for embeddings
- Vector store for storing conversation context
- Streamlit web interface for user interaction

## To Run the app

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/chatbot-with-memory.git
   cd chatbot-wstreamlit run app.pyith-memory
2. **Install the dependecies**:
    ```sh
    pip install -r requirements.txt
3. **Set up environment variables**:
   Create a .env file in the project directory and add your Google Vertex AI API key:
   ```env
   GEMINI_API_KEY=your_google_vertexai_api_key
   ```
4. **Run the Streamlit app**:
   ```sh
   streamlit run app.py
   ```


     

