# backend/tests/test_rag_system.py
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Note: rag_system uses relative imports, ensure they work with proper patching
from rag_system import RAGSystem
from search_tools import CourseSearchTool
from vector_store import SearchResults

class TestRAGSystem(unittest.TestCase):

    @patch('vector_store.VectorStore')
    @patch('rag_system.AIGenerator')
    def setUp(self, mock_ai_generator_class, mock_vector_store_class):
        # Mock dependencies
        self.mock_search_tool = MagicMock(spec=CourseSearchTool)
        self.mock_search_tool.execute.return_value = "Search results found"
        self.mock_search_tool.get_tool_definition.return_value = {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
        self.mock_generator = MagicMock()

        # Configure mocked classes
        self.mock_vector_store_instance = MagicMock()
        mock_vector_store_class.return_value = self.mock_vector_store_instance

        mock_ai_generator_class.return_value = self.mock_generator

        # Mock config for RAGSystem
        self.mock_config = MagicMock()
        self.mock_config.CHUNK_SIZE = 1000
        self.mock_config.CHUNK_OVERLAP = 200
        self.mock_config.CHROMA_PATH = "./chroma_db"
        self.mock_config.MAX_RESULTS = 5
        self.mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.mock_config.OPENROUTER_API_KEY = None
        self.mock_config.ANTHROPIC_API_KEY = "test_key"
        self.mock_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.mock_config.MAX_HISTORY = 5

        # Create the system - this should use our mocked AIGenerator
        self.system = RAGSystem(self.mock_config)

        # Verify that our mocked AIGenerator was called
        mock_ai_generator_class.assert_called_once()

        # Replace the initialized tools with our mocks
        self.system.search_tool = self.mock_search_tool
        # Properly set up the tool manager with our mock tool
        self.system.tool_manager = MagicMock()
        self.system.tool_manager.get_tool_definitions.return_value = [self.mock_search_tool.get_tool_definition()]
        self.system.tool_manager.execute_tool.return_value = "Search results found"
        self.system.tool_manager.get_last_sources.return_value = ["Test Source: Lesson 3"]
        self.system.tool_manager.reset_sources.return_value = None

    def test_content_query_handling(self):
        """Test content queries use search tool"""
        # First response indicates tool use - return a mock that looks like Anthropic tool use response
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_response.content = [
            MagicMock(
                type="tool_use",
                name="search_course_content",
                input={"query": "Python course details"},
                id="tool_123"
            )
        ]

        # Final response after tool execution - return plain text
        final_response = MagicMock()
        final_response.content = [MagicMock()]
        final_response.content[0].text = "Test content results"

        # Configure the mock to return tool use first, then final response
        self.mock_generator.generate_response.side_effect = [tool_use_response, final_response]

        response, sources = self.system.query("Find course details about Python")
        # Debug what we actually got
        print(f"DEBUG: Response type: {type(response)}, value: {response}")
        # The response should be the text from final_response, not the MagicMock itself
        self.assertEqual(response, "Test content results")
        # Verify the search tool was called via tool manager
        self.system.tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python course details"
        )

    def test_general_knowledge_query(self):
        """Test general questions bypass search tool"""
        # For general knowledge, AI should respond directly without tool use - return plain text
        direct_response = MagicMock()
        direct_response.stop_reason = "end_turn"
        direct_response.content = [MagicMock()]
        direct_response.content[0].text = "General knowledge answer"

        self.mock_generator.generate_response.return_value = direct_response

        response, sources = self.system.query("What is the capital of France?")
        print(f"DEBUG: Response type: {type(response)}, value: {response}")
        self.assertEqual(response, "General knowledge answer")
        # Verify search tool was NOT called for general knowledge
        self.system.tool_manager.execute_tool.assert_not_called()

    def test_tool_integration(self):
        """Test tool integration with AI generator"""
        # First response indicates tool use for lesson query
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_response.content = [
            MagicMock(
                type="tool_use",
                name="search_course_content",
                input={"query": "lesson 3 MCP course", "lesson_number": 3},
                id="tool_456"
            )
        ]

        # Final response after tool execution - return plain text
        final_response = MagicMock()
        final_response.content = [MagicMock()]
        final_response.content[0].text = "Search tool result"

        # Configure the mock to return tool use first, then final response
        self.mock_generator.generate_response.side_effect = [tool_use_response, final_response]

        response, sources = self.system.query("What's in lesson 3 of MCP course?")
        print(f"DEBUG: Response type: {type(response)}, value: {response}")
        self.assertEqual(response, "Search tool result")
        # Verify the search tool was called via tool manager
        self.system.tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="lesson 3 MCP course",
            lesson_number=3
        )

    def test_failure_handling(self):
        """Test error propagation"""
        # Simulate search tool returning an error
        self.system.tool_manager.execute_tool.return_value = "Query failed"

        # First response indicates tool use
        tool_use_response = MagicMock()
        tool_use_response.stop_reason = "tool_use"
        tool_use_response.content = [
            MagicMock(
                type="tool_use",
                name="search_course_content",
                input={"query": "Broken query"},
                id="tool_789"
            )
        ]

        # Final response after tool execution (containing the error) - return plain text
        final_response = MagicMock()
        final_response.content = [MagicMock()]
        final_response.content[0].text = "Query failed"

        # Configure the mock to return tool use first, then final response
        self.mock_generator.generate_response.side_effect = [tool_use_response, final_response]

        response, sources = self.system.query("Broken query")
        print(f"DEBUG: Response type: {type(response)}, value: {response}")
        self.assertIn("Query failed", response)

if __name__ == '__main__':
    unittest.main()