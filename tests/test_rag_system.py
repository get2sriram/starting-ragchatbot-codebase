"""Test RAG system initialization"""
import os
import sys
import tempfile

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from config import Config
from rag_system import RAGSystem

def test_rag_system_initialization():
    """Test that RAG system initializes correctly"""
    config = Config()
    # Use a temporary directory for ChromaDB to avoid conflicts
    config.CHROMA_PATH = tempfile.mkdtemp()

    rag_system = RAGSystem(config)
    assert rag_system is not None
    assert rag_system.config == config

if __name__ == "__main__":
    test_rag_system_initialization()
    print("RAG system test passed!")