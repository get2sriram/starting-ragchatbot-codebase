# backend/tests/test_ai_generator.py
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager

class TestAIGenerator(unittest.TestCase):

    def setUp(self):
        # Use a real model that exists
        self.generator = AIGenerator("test_key", "claude-3-sonnet-20240229")

        # Create mock tool and tool manager
        self.mock_tool = MagicMock(spec=CourseSearchTool)
        self.mock_tool.get_tool_definition.return_value = {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }
        self.mock_tool.execute.return_value = "Search results about Python lesson"

        self.tool_manager = MagicMock(spec=ToolManager)
        self.tool_manager.execute_tool.return_value = "Search results about Python lesson"
        self.tool_manager.get_tool_definitions.return_value = [self.mock_tool.get_tool_definition()]

    @patch.object(AIGenerator, '_create')
    def test_tool_usage_when_course_specific(self, mock_create):
        """Test that course-specific questions trigger tool usage"""
        # Mock the tool definition
        tool_def = {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }

        # Simulate the initial response that indicates tool use
        initial_response = MagicMock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [
            MagicMock(
                type="tool_use",
                name="search_course_content",
                input={"query": "MCP course lesson 3"},
                id="tool_123"
            )
        ]

        # Simulate the final response after tool execution
        final_response = MagicMock()
        final_response.content = [MagicMock(text="Answer based on search results.")]

        # Configure mock to return initial then final
        mock_create.side_effect = [initial_response, final_response]

        response = self.generator.generate_response(
            "What is in lesson 3 of MCP course?",
            tools=[tool_def],
            tool_manager=self.tool_manager
        )

        # Tool manager should have been called to execute the tool
        self.tool_manager.execute_tool.assert_called_once()
        self.assertIn("Answer", response)

    @patch.object(AIGenerator, '_create')
    def test_no_tool_for_general_question(self, mock_create):
        """Test that general questions don't trigger tools"""
        # Mock a direct response without tool use
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="Python is a programming language.")]
        mock_create.return_value = mock_response

        tool_def = {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }

        response = self.generator.generate_response(
            "What is Python?",
            tools=[tool_def],
            tool_manager=self.tool_manager
        )

        # Should not call tool for general knowledge
        self.tool_manager.execute_tool.assert_not_called()
        self.assertIn("Python", response)

    @patch.object(AIGenerator, '_create')
    def test_tool_conversion_for_openai(self, mock_create):
        """Test that tools are converted to OpenAI format when using OpenRouter"""
        # Create generator with OpenAI provider (OpenRouter)
        openai_generator = AIGenerator(
            "test_key",
            "gpt-4",
            base_url="https://openrouter.ai/api/v1"
        )

        # Verify provider is set to openai
        self.assertEqual(openai_generator.provider, "openai")

        # Test tool conversion
        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {"type": "object", "properties": {}, "required": []}
        }]

        converted = openai_generator._convert_tools(tools)

        # Verify conversion to OpenAI format
        self.assertEqual(converted[0]["type"], "function")
        self.assertEqual(converted[0]["function"]["name"], "search_course_content")

if __name__ == '__main__':
    unittest.main()