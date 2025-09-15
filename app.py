#!/usr/bin/env python3
"""
RAG Application - Streamlit UI for Querying and Summarizing Research Papers

This Streamlit app provides a user interface for querying the indexed research papers
and generating summaries using a local LLM.

Usage:
    streamlit run app.py

Requirements:
    - Run ingest_and_index.py first to build the index
    - Install dependencies: pip install -r requirements.txt
"""

import os
import json
import logging
from typing import List, Dict, Any
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG system for querying and generating answers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.llm_pipeline = None
        self.index = None
        self.metadata = []
    
    def load_embedding_model(self):
        """Load the sentence transformer model."""
        if self.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = SentenceTransformer(self.model_name)
    
    def load_llm(self):
        """Load the local LLM for text generation."""
        if self.llm_pipeline is None:
            with st.spinner("Loading language model..."):
                # Use a small, efficient model that can run locally
                model_name = "microsoft/DialoGPT-medium"  # Small model for demo
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add padding token if it doesn't exist
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata."""
        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using FAISS."""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Retrieve chunks and metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                chunk_data = self.metadata[idx].copy()
                chunk_data['similarity_score'] = float(score)
                results.append(chunk_data)
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved chunks and local LLM."""
        if self.llm_pipeline is None:
            self.load_llm()
        
        # Create context from retrieved chunks
        context = "\n\n".join([
            f"Source: {chunk['filename']} (Page {chunk['page_number']})\n{chunk['text']}"
            for chunk in retrieved_chunks
        ])
        
        # Create prompt for the LLM
        prompt = f"""Based on the following research paper excerpts, please answer the question: "{query}"

Context:
{context}

Answer (provide a clear, structured response with bullet points and cite sources):"""

        # Generate response
        try:
            response = self.llm_pipeline(
                prompt,
                max_length=len(prompt.split()) + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}"

def format_sources(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks as sources."""
    sources = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = f"{i}. {chunk['filename']} (Page {chunk['page_number']}) - Score: {chunk['similarity_score']:.3f}"
        sources.append(source)
    return "\n".join(sources)

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Research Paper RAG Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Research Paper RAG Assistant")
    st.markdown("Query and summarize research papers using AI-powered retrieval and generation.")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Check if index exists
        index_exists = os.path.exists("faiss_index.bin") and os.path.exists("metadata.json")
        
        if not index_exists:
            st.error("❌ Index not found!")
            st.markdown("""
            **To get started:**
            1. Add PDF files to the `papers/` directory
            2. Run: `python ingest_and_index.py`
            3. Refresh this page
            """)
            st.stop()
        
        st.success("✅ Index loaded successfully!")
        
        # Load index if not already loaded
        if st.session_state.rag_system.index is None:
            with st.spinner("Loading index..."):
                success = st.session_state.rag_system.load_index("faiss_index.bin", "metadata.json")
                if not success:
                    st.error("Failed to load index!")
                    st.stop()
        
        # Configuration options
        k_chunks = st.slider("Number of chunks to retrieve", min_value=3, max_value=10, value=5)
        
        st.markdown("---")
        st.markdown("**Index Statistics:**")
        if st.session_state.rag_system.metadata:
            st.write(f"Total chunks: {len(st.session_state.rag_system.metadata)}")
            unique_files = len(set(chunk['filename'] for chunk in st.session_state.rag_system.metadata))
            st.write(f"Unique files: {unique_files}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask a Question")
        
        # Query input
        query = st.text_area(
            "Enter your question about the research papers:",
            height=100,
            placeholder="e.g., What are the main findings about machine learning in healthcare?"
        )
        
        if st.button("🔍 Search and Generate Answer", type="primary"):
            if not query.strip():
                st.warning("Please enter a question!")
            else:
                with st.spinner("Searching and generating answer..."):
                    # Search for similar chunks
                    retrieved_chunks = st.session_state.rag_system.search_similar_chunks(query, k_chunks)
                    
                    if retrieved_chunks:
                        # Generate answer
                        answer = st.session_state.rag_system.generate_answer(query, retrieved_chunks)
                        
                        # Display results
                        st.subheader("🤖 Generated Answer")
                        st.markdown(answer)
                        
                        # Display sources
                        st.subheader("📚 Sources")
                        sources_text = format_sources(retrieved_chunks)
                        st.text(sources_text)
                    else:
                        st.error("No relevant chunks found. Try rephrasing your question.")
    
    with col2:
        st.header("Retrieved Passages")
        
        if 'retrieved_chunks' in locals():
            for i, chunk in enumerate(retrieved_chunks, 1):
                with st.expander(f"Passage {i} - {chunk['filename']} (Page {chunk['page_number']})"):
                    st.write(f"**Similarity Score:** {chunk['similarity_score']:.3f}")
                    st.write(f"**Text:**")
                    st.write(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Add PDF research papers to the `papers/` directory
    2. Run `python ingest_and_index.py` to build the index
    3. Ask questions about the papers using natural language
    4. The system will retrieve relevant passages and generate answers
    """)

if __name__ == "__main__":
    main()
