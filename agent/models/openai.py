import os
import json
import tiktoken
from openai import OpenAI
from agent.utils import count_tokens, anthropic_to_openai


class OpenAIModel:
    def __init__(self, system_prompt, all_tools, model_name="gpt-4o", max_tokens=4096):
        self.oai_client = OpenAI(api_key=os.getenv("OPENAI"))
        self.model_name = model_name
        self.response_max_tokens = max_tokens
        self.context_window = 124000
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.oai_tools = [anthropic_to_openai(tool) for tool in all_tools]
        self.last_tool_call_id = None

        self.messages = [{"role": "system", "content": system_prompt}]

    def initial_request(self, user_message):
        """Send the first user message (the task). Returns 5-tuple."""
        self.messages.append({"role": "user", "content": user_message})
        return self._send_and_process()

    def send_tool_result(self, content):
        """Send tool result for the last tool call. Returns 5-tuple."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": self.last_tool_call_id,
            "content": content,
        })
        return self._send_and_process()

    def send_user_message(self, content):
        """Send a follow-up user message (e.g. nudge). Returns 5-tuple."""
        self.messages.append({"role": "user", "content": content})
        return self._send_and_process()

    def _send_and_process(self):
        self._truncate_if_needed()

        response = self.oai_client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            tools=self.oai_tools,
            parallel_tool_calls=False,
        )

        # Append assistant message to conversation history
        assistant_message = response.choices[0].message
        self.messages.append(assistant_message.model_dump())

        # Parse the response
        tool_name, response_data = self._parse_response(assistant_message)

        prompt_tokens = response.usage.prompt_tokens
        response_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return tool_name, response_data, total_tokens, prompt_tokens, response_tokens

    def _parse_response(self, message):
        """Extract tool call or text from assistant message. Sets self.last_tool_call_id."""
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            self.last_tool_call_id = tool_call.id
            parsed_args = json.loads(tool_call.function.arguments)
            return tool_call.function.name, parsed_args

        # Text response, no tool call
        return None, message.content

    def _estimate_tokens(self):
        """Approximate total tokens across all messages."""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            elif content is None:
                content = ""
            total += len(self.encoding.encode(str(content), disallowed_special=()))
            # Tool calls in assistant messages
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += len(self.encoding.encode(json.dumps(tool_calls), disallowed_special=()))
        return total

    def _truncate_if_needed(self):
        budget = self.context_window - self.response_max_tokens - 500
        total = self._estimate_tokens()

        if total <= budget:
            return

        # Keep messages[0] (system) and messages[1] (initial user message)
        # Remove pairs from the middle, preserve last 8 messages.
        # Must remove in pairs to maintain OpenAI's invariant:
        # every assistant message with tool_calls must be followed by its tool result.
        preserve_start = 2
        min_tail = 8

        while total > budget and len(self.messages) > preserve_start + min_tail + 1:
            for _ in range(2):
                removed = self.messages.pop(preserve_start)
                content = removed.get("content", "") or ""
                if isinstance(content, list):
                    content = json.dumps(content)
                total -= len(self.encoding.encode(str(content), disallowed_special=()))
                tool_calls = removed.get("tool_calls")
                if tool_calls:
                    total -= len(self.encoding.encode(json.dumps(tool_calls), disallowed_special=()))

        return
