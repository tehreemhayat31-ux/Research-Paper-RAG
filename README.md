# Research Paper RAG Assistant

A Retrieval-Augmented Generation (RAG) application for summarizing and querying research papers using local AI models without requiring paid API keys.

## Features

- **PDF Processing**: Extract text from multiple PDF research papers
- **Intelligent Chunking**: Split text into overlapping chunks for better retrieval
- **Vector Search**: Use FAISS for fast similarity search with sentence transformers
- **Local LLM**: Generate answers using Hugging Face transformers (no API keys needed)
- **Streamlit UI**: Clean, interactive web interface for querying papers
- **Source Citations**: Track and display sources for all generated answers

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Research Papers

Place your PDF research papers in the `papers/` directory:

```bash
# Example: Copy PDFs to papers directory
cp /path/to/your/papers/*.pdf papers/
```

### 3. Build the Index

Process PDFs and create the searchable index:

```bash
python ingest_and_index.py
```

This will:
- Extract text from all PDFs in the `papers/` directory
- Split text into chunks with overlap
- Generate embeddings using sentence-transformers
- Build a FAISS index for fast similarity search
- Save the index and metadata to disk

### 4. Launch the Web App

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser to the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. **Ask Questions**: Enter natural language questions about your research papers
2. **Adjust Retrieval**: Use the sidebar to control how many chunks to retrieve
3. **View Sources**: See which papers and pages the answers come from
4. **Explore Passages**: Click on retrieved passages to see the full text

## Project Structure

```
esearch/
├── papers/                 # Place your PDF files here
├── venv/                  # Virtual environment
├── ingest_and_index.py    # PDF processing and indexing script
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── faiss_index.bin        # Generated FAISS index (after running ingest)
├── metadata.json          # Generated metadata (after running ingest)
└── README.md             # This file
```

## Technical Details

### Dependencies

- **streamlit**: Web UI framework
- **sentence-transformers**: For generating text embeddings
- **faiss-cpu**: Vector database for similarity search
- **PyPDF2**: PDF text extraction
- **transformers**: Hugging Face models for text generation
- **torch**: PyTorch for model inference

### Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional)
- **Text Generation**: `microsoft/DialoGPT-medium` (local, no API keys)

### Configuration

- **Chunk Size**: 1000 characters with 200 character overlap
- **Retrieval**: Top-k similar chunks (adjustable in UI)
- **Generation**: Temperature 0.7 for balanced creativity/accuracy

## Troubleshooting

### Common Issues

1. **"Index not found" error**: Run `python ingest_and_index.py` first
2. **No PDFs found**: Ensure PDF files are in the `papers/` directory
3. **Memory issues**: The local LLM requires ~2GB RAM; consider using a smaller model
4. **Slow performance**: The first run downloads models; subsequent runs are faster

### Performance Tips

- Use smaller PDFs for faster processing
- Adjust chunk size in `ingest_and_index.py` if needed
- Close other applications to free up memory for the LLM

## Customization

### Change Chunk Size

Edit `ingest_and_index.py`:

```python
pdf_processor = PDFProcessor(chunk_size=800, chunk_overlap=150)
```

### Use Different Models

Edit `app.py`:

```python
# For embeddings
rag_system = RAGSystem(model_name="sentence-transformers/all-mpnet-base-v2")

# For text generation (in load_llm method)
model_name = "gpt2"  # or any other Hugging Face model
```

## License

This project is open source and available under the MIT License.
