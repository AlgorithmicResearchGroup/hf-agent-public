import json
import tiktoken

from agent.utils import anthropic_to_openai, ensure_litellm_env


class LiteLLMModel:
    def __init__(self, system_prompt, all_tools, model_name="openai/gpt-5.2", max_tokens=4096):
        from litellm import completion

        ensure_litellm_env()
        self.completion = completion
        self.model_name = model_name
        self.response_max_tokens = max_tokens
        self.context_window = 200000
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.oai_tools = [anthropic_to_openai(tool) for tool in all_tools]
        self.last_tool_call_id = None

        self.messages = [{"role": "system", "content": system_prompt}]

    def initial_request(self, user_message):
        self.messages.append({"role": "user", "content": user_message})
        return self._send_and_process()

    def send_tool_result(self, content):
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": self.last_tool_call_id,
                "content": content,
            }
        )
        return self._send_and_process()

    def send_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        return self._send_and_process()

    def _send_and_process(self):
        self._truncate_if_needed()
        response = self._completion_with_retry()

        assistant_message = response.choices[0].message
        self.messages.append(self._message_to_dict(assistant_message))

        tool_name, response_data = self._parse_response(assistant_message)

        usage = response.usage
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            response_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + response_tokens)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            response_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + response_tokens)

        return tool_name, response_data, total_tokens, prompt_tokens, response_tokens

    def _completion_with_retry(self):
        retry_instruction = (
            "Your previous tool call was rejected because the tool arguments were not valid JSON. "
            "Retry with strict JSON-safe arguments only. "
            "If you are writing a large file, split it into smaller write_file/edit_file calls. "
            "Do not start file content with triple-quoted docstrings, and avoid very large multiline payloads in one tool call."
        )
        for attempt in range(2):
            try:
                return self.completion(
                    model=self.model_name,
                    messages=self.messages,
                    tools=self.oai_tools,
                    parallel_tool_calls=False,
                    max_tokens=self.response_max_tokens,
                    temperature=0,
                )
            except Exception as exc:
                if attempt == 1 or not self._is_json_tool_error(exc):
                    raise
                self.messages.append({"role": "user", "content": retry_instruction})

    def _is_json_tool_error(self, exc):
        text = str(exc)
        return "Failed to parse tool call arguments as JSON" in text or "tool_use_failed" in text

    def _parse_response(self, message):
        tool_calls = self._get_field(message, "tool_calls") or []
        if tool_calls:
            tool_call = tool_calls[0]
            self.last_tool_call_id = self._get_field(tool_call, "id")
            function = self._get_field(tool_call, "function") or {}
            arguments = self._get_field(function, "arguments")
            if isinstance(arguments, str):
                parsed_args = json.loads(arguments)
            else:
                parsed_args = arguments or {}
            return self._get_field(function, "name"), parsed_args

        return None, self._get_field(message, "content")

    def _get_field(self, obj, field):
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    def _message_to_dict(self, message):
        if hasattr(message, "model_dump"):
            return message.model_dump()
        if isinstance(message, dict):
            return message
        return {
            "role": self._get_field(message, "role"),
            "content": self._get_field(message, "content"),
            "tool_calls": self._get_field(message, "tool_calls"),
        }

    def _estimate_tokens(self):
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            elif content is None:
                content = ""
            total += len(self.encoding.encode(str(content), disallowed_special=()))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += len(self.encoding.encode(json.dumps(tool_calls), disallowed_special=()))
        return total

    def _truncate_if_needed(self):
        budget = self.context_window - self.response_max_tokens - 500
        total = self._estimate_tokens()

        if total <= budget:
            return

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
