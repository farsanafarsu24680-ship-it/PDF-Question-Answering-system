# PDF Chatbot RAG System

A complete Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content. The system processes PDFs, creates embeddings, and provides accurate answers based solely on the uploaded document content.

## 🚀 Features

- **PDF Upload & Processing**: Upload any PDF document via the sidebar
- **Text Chunking**: Automatically splits large documents into manageable chunks
- **Embeddings**: Uses sentence-transformers for creating semantic embeddings
- **Vector Search**: FAISS vector store for efficient similarity search
- **RAG Pipeline**: Combines retrieval and generation for accurate answers
- **Chat Interface**: Clean, interactive Streamlit UI with chat history
- **Session Management**: Maintains conversation context
- **Error Handling**: Comprehensive error handling for various scenarios

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS Vector Store
                                                                    ↓
User Question → Retrieval → Relevant Chunks → LLM Generation → Answer
```

### Components:
1. **Frontend**: Streamlit web interface
2. **PDF Processing**: PyPDFLoader for text extraction
3. **Text Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
4. **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
5. **Vector Store**: FAISS for efficient similarity search
6. **LLM**: google/flan-t5-base via Hugging Face Transformers
7. **RAG Chain**: LangChain RetrievalQA

## 📋 Requirements

- Python 3.8+
- 4GB+ RAM (recommended)
- Internet connection (for downloading models on first run)

## 🛠️ Installation & Setup

### 1. Clone/Download the Project
```bash
# Navigate to your project directory
cd C:\Users\farsana\OneDrive\Desktop\qa_system
```

### 2. Activate Virtual Environment
```powershell
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
streamlit run app.py
```

### 5. Access the App
Open your browser and go to: `http://localhost:8501`

## 📖 How to Use

1. **Upload PDF**: Use the sidebar to upload a PDF document
2. **Process Document**: Click "Process PDF" to extract text, create embeddings, and build the vector store
3. **Ask Questions**: Type your questions in the chat interface
4. **Get Answers**: The system will provide answers based only on your PDF content
5. **Continue Chatting**: Ask follow-up questions to explore your document

## 🔧 Configuration

### Model Configuration
The app uses the following models by default:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `google/flan-t5-base`

### Chunking Settings
- Chunk Size: 1000 characters
- Overlap: 200 characters
- Retrieval: Top 3 similar chunks

## 📁 Project Structure

```
qa_system/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── venv/                 # Virtual environment (created)
└── .streamlit/           # Streamlit configuration (auto-created)
```

## 🚨 Important Notes

- **No Hallucinations**: Answers are generated strictly from retrieved PDF chunks
- **CPU-Friendly**: Designed to run on CPU systems with limited RAM
- **Free Models**: Uses only free, open-source models (no API keys required)
- **Privacy**: All processing happens locally on your machine
- **First Run**: Initial model downloads may take several minutes

## 🐛 Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Memory issues**:
   - Close other applications
   - Try smaller PDF files first
   - Use `google/flan-t5-small` instead of `google/flan-t5-base`

3. **Slow processing**:
   - First run downloads models (~500MB)
   - Subsequent runs will be faster

4. **PDF processing fails**:
   - Ensure PDF contains selectable text (not scanned images)
   - Try a different PDF file

## 🔄 Optional Improvements

1. **Model Selection**: Add dropdown to choose between different LLM sizes
2. **Chunk Size Tuning**: Allow users to adjust chunking parameters
3. **Multiple PDFs**: Support uploading multiple documents
4. **Export Chat**: Save conversation history
5. **Document Summary**: Auto-generate document summaries
6. **Citation Display**: Show source page numbers/chunks
7. **Model Caching**: Persistent model storage between sessions

## 📊 Performance Tips

- **Small PDFs**: Process faster and use less memory
- **Clear Chat**: Reset conversation to free memory
- **Close Browser**: Stop Streamlit server when done
- **Virtual Environment**: Keep dependencies isolated

## 🤝 Contributing

This is a complete, production-ready RAG system. Feel free to:
- Report issues
- Suggest improvements
- Fork and modify for your needs

## 📄 License

This project uses open-source components and is free for personal and educational use.

---

**Built with**: Streamlit, LangChain, Hugging Face Transformers, FAISS, and Sentence Transformers