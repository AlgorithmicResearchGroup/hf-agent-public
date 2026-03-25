import os
import json
import tiktoken
import anthropic


class AnthropicModel:
    def __init__(self, system_prompt, all_tools, model_name="claude-sonnet-4-5-20250929", max_tokens=1024):
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC"))
        self.system_prompt = system_prompt
        self.all_tools = all_tools
        self.model_name = model_name
        self.response_max_tokens = max_tokens
        self.context_window = 200000
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.last_tool_call_id = None

        self.messages = []

    def initial_request(self, user_message):
        """Send the first user message (the task). Returns 5-tuple."""
        self.messages.append({"role": "user", "content": user_message})
        return self._send_and_process()

    def send_tool_result(self, content):
        """Send tool result for the last tool call. Returns 5-tuple."""
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": self.last_tool_call_id, "content": content}
            ],
        })
        return self._send_and_process()

    def send_user_message(self, content):
        """Send a follow-up user message (e.g. nudge). Returns 5-tuple."""
        self.messages.append({"role": "user", "content": content})
        return self._send_and_process()

    def _send_and_process(self):
        self._truncate_if_needed()

        response = self.anthropic_client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=self.messages,
            temperature=0,
            max_tokens=self.response_max_tokens,
            tools=self.all_tools,
            tool_choice={"type": "any"},
        )

        # Append assistant message to conversation history
        self.messages.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        })

        # Parse the response
        tool_name, response_data = self._parse_response(response)

        prompt_tokens = response.usage.input_tokens
        response_tokens = response.usage.output_tokens
        total_tokens = prompt_tokens + response_tokens

        return tool_name, response_data, total_tokens, prompt_tokens, response_tokens

    def _parse_response(self, response):
        """Extract tool call or text from response. Sets self.last_tool_call_id."""
        for block in response.content:
            if block.type == "tool_use":
                self.last_tool_call_id = block.id
                return block.name, block.input

        # Text-only response
        for block in response.content:
            if block.type == "text":
                return None, block.text

        raise RuntimeError("Anthropic response contained no content blocks")

    def _estimate_tokens(self):
        """Approximate total tokens across all messages."""
        total = len(self.encoding.encode(self.system_prompt, disallowed_special=()))
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            total += len(self.encoding.encode(str(content), disallowed_special=()))
        return total

    def _truncate_if_needed(self):
        budget = self.context_window - self.response_max_tokens - 500
        total = self._estimate_tokens()

        if total <= budget:
            return

        # Keep messages[0] (initial user message)
        # Remove pairs (assistant + user-tool-result) from the middle, preserve last 8 messages
        preserve_start = 1
        min_tail = 8

        while total > budget and len(self.messages) > preserve_start + min_tail:
            removed = self.messages.pop(preserve_start)
            content = removed.get("content", "") or ""
            if isinstance(content, list):
                content = json.dumps(content)
            total -= len(self.encoding.encode(str(content), disallowed_special=()))

        return
