#!/usr/bin/env python3
"""
RAG Application - PDF Ingestion and Indexing Script

This script processes PDF files from the papers/ directory, extracts text,
creates embeddings using sentence-transformers, and builds a FAISS index.

Usage:
    python ingest_and_index.py

Requirements:
    - Place PDF files in the papers/ directory
    - Install dependencies: pip install -r requirements.txt
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import logging

import PyPDF2
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return list of page texts with metadata."""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        pages.append({
                            'page_number': page_num + 1,
                            'text': text.strip(),
                            'filename': os.path.basename(pdf_path)
                        })
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
        return pages
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Create chunk metadata
            chunk_metadata = {
                'chunk_id': f"{metadata['filename']}_{metadata['page_number']}_{chunk_id}",
                'filename': metadata['filename'],
                'page_number': metadata['page_number'],
                'text': chunk_text,
                'start_char': start,
                'end_char': end
            }
            chunks.append(chunk_metadata)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_id += 1
            
            # Prevent infinite loop
            if start >= len(text) - self.chunk_overlap:
                break
        
        return chunks

class EmbeddingIndexer:
    """Handles embedding generation and FAISS indexing."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
    
    def load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Index loaded from {index_path}")
        logger.info(f"Metadata loaded from {metadata_path}")

def main():
    """Main function to process PDFs and build the index."""
    
    # Configuration
    PAPERS_DIR = "papers"
    INDEX_PATH = "faiss_index.bin"
    METADATA_PATH = "metadata.json"
    
    # Check if papers directory exists and has PDFs
    papers_path = Path(PAPERS_DIR)
    if not papers_path.exists():
        logger.error(f"Papers directory '{PAPERS_DIR}' not found!")
        logger.info("Please create the directory and add PDF files to it.")
        return
    
    pdf_files = list(papers_path.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in '{PAPERS_DIR}' directory!")
        logger.info("Please add PDF files to the papers/ directory.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Initialize processors
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    indexer = EmbeddingIndexer()
    
    # Process all PDFs
    all_chunks = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        logger.info(f"Processing: {pdf_file.name}")
        
        # Extract text from PDF
        pages = pdf_processor.extract_text_from_pdf(str(pdf_file))
        
        # Create chunks from each page
        for page in pages:
            chunks = pdf_processor.chunk_text(page['text'], page)
            all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} text chunks")
    
    if not all_chunks:
        logger.error("No text chunks created! Check your PDF files.")
        return
    
    # Prepare texts and metadata for embedding
    texts = [chunk['text'] for chunk in all_chunks]
    indexer.metadata = all_chunks
    
    # Create embeddings
    embeddings = indexer.create_embeddings(texts)
    
    # Build FAISS index
    indexer.build_faiss_index(embeddings)
    
    # Save index and metadata
    indexer.save_index(INDEX_PATH, METADATA_PATH)
    
    logger.info("Indexing completed successfully!")
    logger.info(f"Total chunks indexed: {len(all_chunks)}")
    logger.info(f"Index file: {INDEX_PATH}")
    logger.info(f"Metadata file: {METADATA_PATH}")

if __name__ == "__main__":
    main()
