# backend/tests/test_search_tool.py
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults

class TestCourseSearchTool(unittest.TestCase):

    def setUp(self):
        # Mock VectorStore with all required fields
        self.mock_vector_store = MagicMock()
        mock_search_result = SearchResults(
            documents=["test document 1", "test document 2"],
            metadata=[{
                'course_title': 'MCP Course',
                'lesson_number': 2
            }],
            error=None,
            distances=[0.1, 0.2]  # Required by SearchResults
        )
        self.mock_vector_store.search.return_value = mock_search_result
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        self.tool = CourseSearchTool(self.mock_vector_store)

    def test_execution_with_query(self):
        """Test basic query execution returns results"""
        result = self.tool.execute(query="test")
        self.assertIn("test document", result)
        self.mock_vector_store.search.assert_called_once()

    def test_execution_with_course_filter(self):
        """Test query with course name filter"""
        result = self.tool.execute(query="lesson", course_name="MCP Course")
        self.assertIn("MCP", result)
        self.mock_vector_store.search.assert_called_with(
            query="lesson",
            course_name="MCP Course",
            lesson_number=None
        )

    def test_empty_results_handling(self):
        """Test handling of empty search results"""
        mock_result = SearchResults(
            documents=[],
            metadata=[],
            error=None,
            distances=[]
        )
        self.mock_vector_store.search.return_value = mock_result
        result = self.tool.execute(query="nothing")
        self.assertIn("No relevant content", result)

    def test_error_handling(self):
        """Test handling of search errors"""
        mock_result = SearchResults(
            documents=[],
            metadata=[],
            error="Database connection failed",
            distances=[]
        )
        self.mock_vector_store.search.return_value = mock_result
        result = self.tool.execute(query="test")
        self.assertEqual(result, "Database connection failed")

if __name__ == '__main__':
    unittest.main()