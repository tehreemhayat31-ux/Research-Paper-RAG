#!/usr/bin/env python3
"""
Test script to verify the RAG application setup.
Run this after installing dependencies to check if everything works.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist."""
    print("\nTesting directories...")
    
    papers_dir = Path("papers")
    if papers_dir.exists():
        print("✅ Papers directory exists")
        
        pdf_files = list(papers_dir.glob("*.pdf"))
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF files")
            for pdf in pdf_files:
                print(f"   - {pdf.name}")
        else:
            print("⚠️  No PDF files found in papers/ directory")
    else:
        print("❌ Papers directory not found")
        return False
    
    return True

def test_scripts():
    """Test if main scripts exist and are executable."""
    print("\nTesting scripts...")
    
    scripts = ["ingest_and_index.py", "app.py"]
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} not found")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 RAG Application Setup Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test directories
    if not test_directories():
        all_passed = False
    
    # Test scripts
    if not test_scripts():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Add PDF files to the papers/ directory")
        print("2. Run: python ingest_and_index.py")
        print("3. Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo fix:")
        print("1. Make sure you're in the virtual environment: source venv/bin/activate")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Create papers directory: mkdir papers")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
