#!/usr/bin/env python3
"""
Test script to evaluate RAG system accuracy
"""

import json
import os
from app import RAGSystem

def test_rag_accuracy():
    """Test the RAG system with sample questions."""
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Load index
    if not rag_system.load_index("faiss_index.bin", "metadata.json"):
        print("❌ Failed to load index!")
        return
    
    print("✅ Index loaded successfully!")
    print(f"📊 Total chunks: {len(rag_system.metadata)}")
    
    # Sample test questions
    test_questions = [
        "What is the main topic of these research papers?",
        "What methodology was used in the studies?",
        "What are the key findings or conclusions?",
        "What data was collected and how?",
        "What are the limitations mentioned?",
        "Who are the authors of these papers?",
        "What is the abstract or summary of the research?",
        "What tools or software were used?",
        "What are the future research directions?",
        "What statistical methods were applied?"
    ]
    
    print("\n🧪 Testing RAG System Accuracy...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        # Search for relevant chunks
        retrieved_chunks = rag_system.search_similar_chunks(question, k=5)
        
        if retrieved_chunks:
            print(f"🔍 Found {len(retrieved_chunks)} relevant chunks")
            
            # Show top chunk similarity scores
            print("📊 Top similarity scores:")
            for j, chunk in enumerate(retrieved_chunks[:3], 1):
                print(f"   {j}. {chunk['filename']} (Page {chunk['page_number']}) - Score: {chunk['similarity_score']:.3f}")
            
            # Generate answer
            print("🤖 Generating answer...")
            answer = rag_system.generate_answer(question, retrieved_chunks)
            print(f"💡 Answer: {answer[:200]}...")
            
        else:
            print("❌ No relevant chunks found")
        
        print("-" * 30)
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_rag_accuracy()
