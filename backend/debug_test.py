#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

from unittest.mock import patch, MagicMock
from rag_system import RAGSystem

print("Testing RAG system query method...")

# Mock the dependencies
with patch('vector_store.VectorStore') as mock_vs, patch('rag_system.AIGenerator') as mock_ai:
    mock_vs_instance = MagicMock()
    mock_vs.return_value = mock_vs_instance
    mock_ai_instance = MagicMock()
    mock_ai.return_value = mock_ai_instance

    # Mock config
    mock_config = MagicMock()
    mock_config.CHUNK_SIZE = 1000
    mock_config.CHUNK_OVERLAP = 200
    mock_config.CHROMA_PATH = './chroma_db'
    mock_config.MAX_RESULTS = 5
    mock_config.EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    mock_config.OPENROUTER_API_KEY = None
    mock_config.ANTHROPIC_API_KEY = 'test_key'
    mock_config.ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'
    mock_config.MAX_HISTORY = 5

    # Create system
    system = RAGSystem(mock_config)

    # Replace with our mocks
    system.search_tool = MagicMock()
    system.search_tool.execute.return_value = 'Test results'
    system.tool_manager = MagicMock()
    system.tool_manager.get_tool_definitions.return_value = []
    system.tool_manager.get_last_sources.return_value = []
    system.tool_manager.reset_sources.return_value = None

    # Mock the AI generator response to return a simple string (what we expect)
    mock_ai_instance.generate_response.return_value = 'Direct answer from AI'

    # Test query
    response, sources = system.query('test query')
    print(f'Response: {response}')
    print(f'Response type: {type(response)}')
    print(f'Sources: {sources}')

    # Now test with tool use scenario
    print("\\nTesting tool use scenario...")

    # Mock tool use response
    tool_use_response = MagicMock()
    tool_use_response.stop_reason = "tool_use"
    tool_use_response.content = [
        MagicMock(
            type="tool_use",
            name="search_course_content",
            input={"query": "test"},
            id="tool_123"
        )
    ]

    # Final response
    final_response = MagicMock()
    final_response.content = [MagicMock()]
    final_response.content[0].text = "Final answer after tool use"

    mock_ai_instance.generate_response.side_effect = [tool_use_response, final_response]

    # Set up tool manager to return search results
    system.tool_manager.execute_tool.return_value = "Search results from tool"
    system.tool_manager.get_last_sources.return_value = ["Test Source"]

    response2, sources2 = system.query('test query with tools')
    print(f'Response with tools: {response2}')
    print(f'Response type: {type(response2)}')
    print(f'Sources: {sources2}')