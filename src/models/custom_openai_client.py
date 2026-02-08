"""Generic OpenAI-compatible client for any provider that exposes the /v1/chat/completions API.

Supports: DeepSeek, Mistral, Groq, Together, Fireworks, xAI/Grok, Cerebras,
Azure OpenAI, and any other service that speaks the OpenAI chat completions
format.

Configure via environment variables:
    AGENTARMY_CUSTOM_API_KEY       — API key
    AGENTARMY_CUSTOM_API_BASE      — Base URL (e.g. https://api.deepseek.com/v1)
    AGENTARMY_CUSTOM_DEFAULT_MODEL — Model name (e.g. deepseek-coder)
"""

import asyncio
import json
import os
import time
from typing import Any, AsyncGenerator, Optional

import httpx
import structlog

from src.models.schemas import LLMRequest, LLMResponse, ToolCall, ModelProvider

logger = structlog.get_logger(__name__)


class CustomOpenAIClient:
    """Async client for any OpenAI-compatible LLM provider.

    Supports:
    - Async chat completion (httpx-based, zero extra dependencies)
    - Streaming responses (SSE)
    - Token counting (approximate)
    - Exponential backoff retry logic
    - Configurable base URL, model, and API key
    """

    # Generic fallback pricing (per 1M tokens) — users won't get exact costs,
    # but tracking still works and gives a rough order-of-magnitude estimate.
    DEFAULT_PRICING = {"input": 1.0, "output": 2.0}

    REQUEST_TIMEOUT = 120.0  # 2 minutes

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        api_base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: float = REQUEST_TIMEOUT,
    ) -> None:
        """Initialize the generic OpenAI-compatible client.

        Args:
            api_key: API key for the provider.
            model: Model identifier (provider-specific). Falls back to env
                   AGENTARMY_CUSTOM_DEFAULT_MODEL or "gpt-3.5-turbo".
            api_base_url: Base URL of the OpenAI-compatible API. Falls back
                          to env AGENTARMY_CUSTOM_API_BASE or OpenAI default.
            max_retries: Maximum number of retries for failed requests.
            timeout_seconds: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = (
            model
            or os.environ.get("AGENTARMY_CUSTOM_DEFAULT_MODEL")
            or "gpt-3.5-turbo"
        )
        self.api_base_url = (
            api_base_url
            or os.environ.get("AGENTARMY_CUSTOM_API_BASE")
            or "https://api.openai.com/v1"
        )
        # Strip trailing slash for consistent path joining
        self.api_base_url = self.api_base_url.rstrip("/")

        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self.client = httpx.AsyncClient(
            base_url=self.api_base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "CustomOpenAIClient initialized",
            model=self.model,
            base_url=self.api_base_url,
            max_retries=max_retries,
        )

    # ── Public interface ─────────────────────────────────────────

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute an async chat completion request.

        Args:
            request: The LLM request with prompt and parameters.

        Returns:
            LLMResponse with the model's response and metadata.

        Raises:
            Exception: If all retry attempts fail.
        """
        start_time = time.time()

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        # Build payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Tool calling (if the provider supports it)
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"

        # Retry with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "custom_openai_request",
                    model=self.model,
                    base_url=self.api_base_url,
                    attempt=attempt + 1,
                )

                response = await self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                # Parse response
                content_text = ""
                tool_calls: list[ToolCall] = []
                finish_reason = "unknown"

                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    finish_reason = choice.get("finish_reason", "unknown")

                    if message.get("content"):
                        content_text = message["content"]

                    # Tool calls
                    for tc in message.get("tool_calls", []):
                        if tc.get("type") == "function":
                            func = tc.get("function", {})
                            arguments: dict[str, Any] = {}
                            if "arguments" in func:
                                try:
                                    arguments = json.loads(func["arguments"])
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            tool_calls.append(
                                ToolCall(
                                    tool_name=func.get("name", "unknown"),
                                    arguments=arguments,
                                    id=tc.get("id"),
                                )
                            )

                # Token usage
                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = input_tokens + output_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                logger.info(
                    "custom_openai_success",
                    model=self.model,
                    tokens=total_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                )

                return LLMResponse(
                    content=content_text,
                    model_used=self.model,
                    provider=ModelProvider.CUSTOM,
                    tokens_used=total_tokens,
                    cost_estimate=cost,
                    latency_ms=latency_ms,
                    tool_calls=tool_calls if tool_calls else None,
                    stop_reason=finish_reason,
                )

            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                    wait = (2 ** attempt) + (time.time() % 1)
                    logger.warning(
                        "custom_openai_retry",
                        status=status,
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("custom_openai_http_error", status=status, error=str(exc))
                raise

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "custom_openai_error",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        logger.error(
            "custom_openai_all_retries_failed",
            model=self.model,
            retries=self.max_retries,
        )
        raise last_error  # type: ignore[misc]

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Yields text chunks via SSE as they arrive.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue

        except Exception as exc:
            logger.error("custom_openai_stream_error", error=str(exc))
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count (1 token ~ 4 characters).

        Args:
            text: Text to estimate tokens for.

        Returns:
            Approximate token count.
        """
        return max(1, len(text) // 4)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
        logger.debug("CustomOpenAIClient closed")

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            asyncio.run(self.close())
        except Exception:
            pass

    # ── Private ──────────────────────────────────────────────────

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost using generic pricing.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = self.DEFAULT_PRICING
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
