# Import Streamlit for creating the web interface
import streamlit as st

# Import os for file operations (removing temporary files)
import os

# Import tempfile for creating temporary files during PDF processing
import tempfile

# Import PyPDFLoader from LangChain to load and extract text from PDF files
from langchain_community.document_loaders import PyPDFLoader

# Import RecursiveCharacterTextSplitter for splitting long documents into manageable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import HuggingFaceEmbeddings to create vector embeddings from text
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import FAISS vector store for efficient similarity search
from langchain_community.vectorstores import FAISS

# Import transformers components for loading the LLM tokenizer and model
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Import torch for PyTorch operations (used by transformers)
import torch

# Import numpy for numerical operations (used by sentence transformers)
import numpy as np

# Configure the Streamlit page appearance and layout
st.set_page_config(
    page_title="PDF Question Answering System",  # Title shown in browser tab
    page_icon="📚",  # Emoji icon for the browser tab
    layout="wide"  # Use wide layout for better space utilization
)

# No custom CSS needed - using Streamlit's built-in components and styling

def initialize_session_state():
    """Initialize all session state variables used throughout the application"""
    if 'chat_history' not in st.session_state:  # Check if chat_history doesn't exist in session
        st.session_state.chat_history = []  # Initialize as empty list to store conversation

    if 'vectorstore' not in st.session_state:  # Check if vectorstore doesn't exist
        st.session_state.vectorstore = None  # Initialize as None, will hold FAISS vector store

    if 'tokenizer' not in st.session_state:  # Check if tokenizer doesn't exist
        st.session_state.tokenizer = None  # Initialize as None, will hold the LLM tokenizer

    if 'model' not in st.session_state:  # Check if model doesn't exist
        st.session_state.model = None  # Initialize as None, will hold the LLM model

    if 'pdf_processed' not in st.session_state:  # Check if pdf_processed flag doesn't exist
        st.session_state.pdf_processed = False  # Initialize as False, tracks if PDF is ready

    if 'pdf_name' not in st.session_state:  # Check if pdf_name doesn't exist
        st.session_state.pdf_name = None  # Initialize as None, stores uploaded PDF filename

def process_pdf(uploaded_file):
    """Process uploaded PDF file: extract text, create chunks, embeddings, and vector store"""
    try:
        # Create a temporary file to save the uploaded PDF content for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())  # Write uploaded file content to temp file
            tmp_file_path = tmp_file.name  # Get the path of the temporary file

        # Initialize PyPDFLoader to extract text from the PDF file
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()  # Load and extract text from all PDF pages

        # Remove the temporary file to free up disk space
        os.unlink(tmp_file_path)

        # Check if any text content was extracted from the PDF
        if not documents:
            st.error("No text content found in the PDF. Please try a different PDF.")
            return False  # Return False to indicate processing failure

        # Initialize text splitter to break long documents into smaller, manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Maximum characters per chunk
            chunk_overlap=200,  # Number of overlapping characters between chunks
            length_function=len  # Function to measure chunk length
        )
        chunks = text_splitter.split_documents(documents)  # Split documents into chunks

        # Initialize the embedding model to convert text chunks into vector representations
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Pre-trained embedding model
            model_kwargs={'device': 'cpu'}  # Use CPU for processing (no GPU required)
        )

        # Create FAISS vector store from the document chunks and their embeddings
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Load the pre-trained language model and tokenizer for question answering
        model_name = "google/flan-t5-base"  # Free, efficient T5 model
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Load the model

        # Store all processed components in Streamlit session state for later use
        st.session_state.vectorstore = vectorstore  # Store the vector store
        st.session_state.tokenizer = tokenizer  # Store the tokenizer
        st.session_state.model = model  # Store the language model
        st.session_state.pdf_processed = True  # Mark PDF as processed
        st.session_state.pdf_name = uploaded_file.name  # Store PDF filename

        return True  # Return True to indicate successful processing

    except Exception as e:
        # Display error message to user if PDF processing fails
        st.error(f"Error processing PDF: {str(e)}")
        return False  # Return False to indicate processing failure

def get_response(question):
    """Generate a response using the RAG (Retrieval-Augmented Generation) pipeline"""
    try:
        # Check if PDF has been processed and all required components are available
        if not st.session_state.vectorstore or not st.session_state.tokenizer or not st.session_state.model:
            return "Please upload and process a PDF first.", []  # Return error message and empty list

        # Perform similarity search to find the most relevant document chunks
        docs = st.session_state.vectorstore.similarity_search(question, k=3)  # Get top 3 similar chunks
        context = "\n\n".join([doc.page_content for doc in docs])  # Combine chunk contents

        # Construct a prompt that instructs the model to answer based only on the provided context
        prompt = f"""Answer the question based only on the provided context. If the answer is not available in the context, say "The answer is not available in the provided document."

Context:
{context}

Question: {question}

Answer:"""

        # Tokenize the prompt for model input
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

        # Generate response using the language model with specific parameters for quality
        outputs = st.session_state.model.generate(
            inputs["input_ids"],  # Tokenized input
            max_length=512,  # Maximum length of generated response
            num_beams=4,  # Beam search for better quality (4 candidates)
            early_stopping=True,  # Stop when EOS token is generated
            temperature=0.1,  # Low temperature for more deterministic output
            do_sample=False,  # Disable sampling for consistent results
            repetition_penalty=1.1  # Penalize repetitive text
        )

        # Decode the generated tokens back to text
        answer = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove leading/trailing whitespace from the answer
        answer = answer.strip()

        # Check if the model indicates no relevant information was found
        if "not available" in answer.lower() or "cannot find" in answer.lower():
            answer = "The answer is not available in the provided document."  # Standardized response

        return answer, docs  # Return the answer and source documents

    except Exception as e:
        # Return error message and empty list if something goes wrong
        return f"Error generating response: {str(e)}", []

def main():
    """Main function that defines the Streamlit application layout and logic"""
    initialize_session_state()  # Initialize all required session state variables

    # Display the main page header using Streamlit's built-in title component
    st.title("📚 PDF Question Answering System")  # Main title with emoji
    st.markdown("### Ask questions about your PDF documents using AI!")  # Subtitle description

    # Create sidebar for PDF upload and processing controls
    with st.sidebar:
        st.header("📤 Upload PDF")  # Sidebar header using Streamlit component

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")  # File uploader widget for PDFs

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Display process button when file is uploaded
            if st.button("Process PDF", type="primary"):
                # Show loading spinner during processing (may take time for model downloads)
                with st.spinner("Processing PDF... This may take a few minutes."):
                    success = process_pdf(uploaded_file)  # Process the uploaded PDF
                    if success:  # If processing was successful
                        st.success("✅ PDF processed successfully!")  # Show success message
                        st.info(f"📄 **{st.session_state.pdf_name}** is ready for questions!")  # Show ready message
                    else:  # If processing failed
                        st.error("❌ Failed to process PDF. Please try again.")  # Show error message

        # Display status information when PDF is processed
        if st.session_state.pdf_processed:
            st.markdown("---")  # Add separator line
            st.markdown("### 📋 Status")  # Status section header
            st.success("✅ PDF Loaded & Processed")  # Green success indicator
            st.info(f"📄 Document: {st.session_state.pdf_name}")  # Show document name

            # Button to clear current PDF and reset the application
            if st.button("🗑️ Clear PDF", help="Remove current PDF and start fresh"):
                st.session_state.vectorstore = None  # Clear vector store
                st.session_state.tokenizer = None  # Clear tokenizer
                st.session_state.model = None  # Clear model
                st.session_state.pdf_processed = False  # Reset processed flag
                st.session_state.pdf_name = None  # Clear PDF name
                st.session_state.chat_history = []  # Clear chat history
                st.rerun()  # Refresh the app to reflect changes

    # Main chat interface - only show if PDF is processed
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload and process a PDF file from the sidebar to start chatting.")
        return  # Exit early if no PDF is processed

    # Create a container to hold the chat history display
    chat_container = st.container()

    # Display the conversation history using Streamlit's built-in chat components
    with chat_container:
        for message in st.session_state.chat_history:  # Loop through each message in history
            if message["role"] == "user":  # If message is from the user
                with st.chat_message("user"):  # Use Streamlit's chat message component
                    st.write(message["content"])  # Display user message content
            else:  # If message is from the assistant
                with st.chat_message("assistant"):  # Use Streamlit's chat message component
                    st.write(message["content"])  # Display assistant message content

    # Create input area for user questions
    st.markdown("---")  # Add separator line above input area
    with st.form(key="chat_form", clear_on_submit=True):  # Use form to prevent auto-refresh on submit
        user_question = st.text_input(  # Create text input field
            "Ask a question about your PDF:",  # Label for the input field
            placeholder="What would you like to know about the document?",  # Placeholder text
            key="user_input"  # Unique key for the input widget
        )
        submit_button = st.form_submit_button("Send Question", type="primary")  # Submit button

    # Process user question when submit button is clicked and question is not empty
    if submit_button and user_question.strip():
        # Add the user's question to the chat history for display
        st.session_state.chat_history.append({
            "role": "user",  # Mark as user message
            "content": user_question.strip()  # Store cleaned question text
        })

        # Generate response using RAG pipeline with loading indicator
        with st.spinner("Thinking..."):  # Show loading spinner during processing
            answer, sources = get_response(user_question.strip())  # Get answer from RAG system

        # Add the assistant's response to the chat history
        st.session_state.chat_history.append({
            "role": "assistant",  # Mark as assistant message
            "content": answer  # Store the generated answer
        })

        # Refresh the app to update the chat display with new messages
        st.rerun()

    # Show clear chat button only if there are messages in history
    if st.session_state.chat_history:
        st.markdown("---")  # Add separator line
        col1, col2, col3 = st.columns([1, 1, 1])  # Create 3 equal columns for centering
        with col2:  # Place button in middle column
            if st.button("🧹 Clear Chat History"):  # Button to clear conversation
                st.session_state.chat_history = []  # Reset chat history to empty list
                st.rerun()  # Refresh app to update display

    # Display footer with credits using Streamlit components
    st.divider()  # Add separator line above footer
    st.caption("Built with Streamlit, LangChain, and Hugging Face Transformers")  # Technology credits

# Execute main function when script is run directly (not imported as module)
if __name__ == "__main__":
    main()  # Start the Streamlit application