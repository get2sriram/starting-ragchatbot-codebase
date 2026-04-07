import anthropic
import openai
import time
from typing import List, Optional, Dict, Any

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds


class AIGenerator:
    """Handles interactions with AI APIs for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = (
        "You are an AI assistant specialized in course materials and educational content\n"
        "with access to a comprehensive search tool for course information.\n\n"
        "Search Tool Usage:\n"
        "- Use the search tool **only** for questions about specific course content\n"
        "  or detailed educational materials\n"
        "- **One search per query maximum**\n"
        "- Synthesize search results into accurate, fact-based responses\n"
        "- If search yields no results, state this clearly without offering alternatives\n\n"
        "Response Protocol:\n"
        "- **General knowledge questions**: Answer using existing knowledge without searching\n"
        "- **Course-specific questions**: Search first, then answer\n"
        "- **No meta-commentary**:\n"
        "  - Provide direct answers only — no reasoning process, search explanations,\n"
        "    or question-type analysis\n"
        '  - Do not mention "based on the search results"\n\n'
        "All responses must be:\n"
        "1. **Brief, Concise and focused** - Get to the point quickly\n"
        "2. **Educational** - Maintain instructional value\n"
        "3. **Clear** - Use accessible language\n"
        "4. **Example-supported** - Include relevant examples when they aid understanding\n"
        "Provide only the direct answer to what was asked."
    )

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.model = model

        if base_url:
            # OpenRouter uses OpenAI-compatible API format
            self.provider = "openai"
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        else:
            # Anthropic API
            self.provider = "anthropic"
            self.client = anthropic.Anthropic(api_key=api_key)

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
        }

        if self.provider == "anthropic":
            api_params["system"] = system_content
        else:
            api_params["messages"].insert(
                0, {"role": "system", "content": system_content}
            )

        # Add tools if available
        if tools:
            api_params["tools"] = self._convert_tools(tools)
            if self.provider == "anthropic":
                api_params["tool_choice"] = {"type": "auto"}
            else:
                api_params["tool_choice"] = "auto"

        response = self._create(**api_params)

        # Handle tool execution if needed
        if (
            self.provider == "anthropic"
            and response.stop_reason == "tool_use"
            and tool_manager
        ):
            return self._handle_tool_execution_anthropic(
                response, api_params, tool_manager
            )
        elif (
            self.provider == "openai"
            and response.choices[0].finish_reason == "tool_calls"
            and tool_manager
        ):
            return self._handle_tool_execution_openai(
                response, api_params, tool_manager
            )

        # Return direct response
        if self.provider == "anthropic":
            return response.content[0].text
        return response.choices[0].message.content

    def _create(self, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                if self.provider == "anthropic":
                    return self.client.messages.create(**kwargs)
                return self.client.chat.completions.create(**kwargs)
            except (openai.RateLimitError, anthropic.RateLimitError):
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = INITIAL_RETRY_DELAY * (2**attempt)
                print(
                    f"Rate limited, retrying in {delay}s... "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(delay)

    def _handle_tool_execution_anthropic(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }
        return self._create(**final_params).content[0].text

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-format tools to OpenAI format if needed"""
        if self.provider == "anthropic":
            return tools
        # OpenAI expects: {"type": "function", "function": {
        #   "name": ..., "description": ..., "parameters": {...}}}
        # Anthropic has: {"name": ..., "description": ..., "input_schema": {...}}
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t.get(
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                },
            }
            for t in tools
        ]

    def _handle_tool_execution_openai(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        messages = base_params["messages"].copy()
        messages.append(initial_response.choices[0].message)

        for tool_call in initial_response.choices[0].message.tool_calls:
            tool_result = tool_manager.execute_tool(
                tool_call.function.name, **eval(tool_call.function.arguments)
            )
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result}
            )

        final_params = {**self.base_params, "messages": messages}
        return self._create(**final_params).choices[0].message.content
