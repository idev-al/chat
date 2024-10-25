import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

# Create a Flask application
app = Flask(__name__)

# Global variables to hold the index, loaded documents, conversation history, and the LLM
index = None
documents = None
conversation_history = []  # This list will store the conversation history
llm = None  # Language model for handling queries

# Define the system prompt
SYSTEM_PROMPT = """You are a highly knowledgeable assistant. Provide clear, accurate, and concise responses, while maintaining a polite tone."""

# Function to load documents from a directory
def load_documents(directory_path):
    global documents
    if documents is None:
        # Use SimpleDirectoryReader to load all document types
        documents = SimpleDirectoryReader(directory_path).load_data()

# Function to construct an index from loaded documents
def construct_index(directory_path):
    global index
    load_documents(directory_path)

    # Build the VectorStoreIndex from the loaded documents
    if index is None:
        index = VectorStoreIndex.from_documents(documents)

    # Use a directory path to save the index files
    os.makedirs('index_dir', exist_ok=True)  # Ensure the directory exists
    index.storage_context.persist('index_dir')  # Save the index to a directory

    return index

# Function to load the index from a directory
def load_index():
    # Load the index from the directory
    index = VectorStoreIndex.load('index_dir')
    return index

# Function to set up the OpenAI GPT model
def initialize_llm():
    global llm
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Adjust temperature if needed

# Function for chatbot interaction
def chatbot(input_text):
    global index, llm, conversation_history
    
    if index is None:
        index = VectorStoreIndex.load('index_dir')  # Load the index from the directory
    
    # Append the user input to the conversation history
    conversation_history.append(f"User: {input_text}")
    
    # Query the document index first to get relevant information
    query_engine = index.as_query_engine()
    document_response = query_engine.query(input_text)
    document_context = document_response.response  # Get the response from the document index
    
    # Initialize the GPT model if not done yet
    initialize_llm()

    # Combine the system prompt, document context, and conversation history
    conversation_context = "\n".join(conversation_history)
    full_prompt = f"{SYSTEM_PROMPT}\n\nDocument Context: {document_context}\n\n{conversation_context}"

    # Query the GPT model with the user's input and the document context
    gpt_response = llm(full_prompt)

    # Extract the content from the AIMessage object
    response_content = gpt_response.content

    # Append the chatbot's response to the conversation history
    conversation_history.append(f"{response_content}")
    
    return response_content  # Return only the actual content


# Create a Gradio interface for the chatbot
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        label="Chat",
        placeholder="Type your message...",
        lines=5
    ),
    outputs=gr.Textbox(
        label="Chatbot Response",
        type="text"
    ),
    live=True,
    title="Chatbot with GPT-3.5 Model"
)

# Define an API endpoint for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot_api():
    user_message = request.json.get('user_message')
    response = chatbot(user_message)
    return jsonify({'chatbot_response': response})

if __name__ == '__main__':
    # Launch the Gradio interface
    index = construct_index("docs")
    iface.launch(share=True)
    # Run the Flask app in parallel to expose the API
    app.run(port=5000)  # Adjust the port as needed
