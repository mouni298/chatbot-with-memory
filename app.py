import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the language model and embeddings
llm = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai")
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Initialize the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

retriever = vector_store.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

# Define the prompt template
prompt_template = """The following is a friendly conversation between a user and a chatbot. The chatbot is talkative and provides lots of specific details from its context. If the chatbot does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
User: {input}
Chatbot:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=prompt_template)

# Initialize the conversation chain
conversation_with_memory = ConversationChain(
    llm=llm, prompt=prompt, memory=memory, verbose=True
)

# Define the chat function
def chat_with_memory(user_input):
    answer = conversation_with_memory.predict(input=user_input)
    memory.save_context({"input": user_input}, {"output": answer})
    return answer

# Streamlit app
st.title("Chatbot with Memory")

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        answer = chat_with_memory(user_input)
        st.text_area("Chatbot:", value=answer, height=200, max_chars=None, key=None)