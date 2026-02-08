"""Anthropic Claude integration with async support, tool calling, and token counting."""

import asyncio
import time
from typing import AsyncGenerator, Optional, Any
import structlog

try:
    from anthropic import Anthropic, AsyncAnthropic
except ImportError:
    raise ImportError(
        "anthropic package not found. Install with: pip install anthropic"
    )

from src.models.schemas import LLMRequest, LLMResponse, ToolCall, ModelProvider


logger = structlog.get_logger(__name__)


class ClaudeClient:
    """Async client for Anthropic Claude models with advanced features.

    Supports:
    - Async chat completion
    - Tool use / function calling
    - Streaming responses
    - Token counting and cost tracking
    - Automatic retry logic
    - Conversation memory management
    """

    # Model pricing (per 1M tokens) as of knowledge cutoff
    MODEL_PRICING = {
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.80, "output": 4.0},
        "claude-opus-4-6": {"input": 5.0, "output": 25.0},
        "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    }

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key.
            model: Model to use for requests.
            max_retries: Maximum number of retries for failed requests.
            timeout_seconds: Request timeout in seconds.
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Initialize async client
        self.client = AsyncAnthropic(api_key=api_key)

        # Conversation history for memory management
        self.conversation_history: dict[str, list[dict[str, Any]]] = {}

        logger.info(
            "ClaudeClient initialized",
            model=model,
            max_retries=max_retries,
        )

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

        # Prepare system message if provided
        system = request.system_prompt or ""

        # Prepare messages — multimodal if attachments are present
        if request.attachments:
            content_parts: list[dict] = [{"type": "text", "text": request.prompt}]
            for block in request.attachments:
                if block.type == "image" and block.data and block.media_type:
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data,
                        },
                    })
                elif block.type == "document" and block.text:
                    label = f"[File: {block.filename}]" if block.filename else "[Attached file]"
                    content_parts.append({"type": "text", "text": f"{label}\n{block.text}"})
                elif block.type == "text" and block.text:
                    content_parts.append({"type": "text", "text": block.text})
            messages = [{"role": "user", "content": content_parts}]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        # Convert tools format if provided
        tools_param = None
        if request.tools:
            tools_param = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    "Sending request to Claude",
                    model=self.model,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )

                # Build kwargs — only include optional params when set
                create_kwargs = {
                    "model": self.model,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "messages": messages,
                }
                if system:
                    create_kwargs["system"] = system
                if tools_param:
                    create_kwargs["tools"] = tools_param

                response = await self.client.messages.create(**create_kwargs)

                # Process response
                latency_ms = (time.time() - start_time) * 1000

                # Extract text content and tool calls
                content_text = ""
                tool_calls = []

                for block in response.content:
                    if hasattr(block, "text"):
                        content_text += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                tool_name=block.name,
                                arguments=block.input,
                                id=block.id,
                            )
                        )

                # Calculate cost
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                total_tokens = input_tokens + output_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                logger.info(
                    "Claude request successful",
                    model=self.model,
                    tokens_used=total_tokens,
                    cost=cost,
                    latency_ms=latency_ms,
                )

                return LLMResponse(
                    content=content_text,
                    model_used=self.model,
                    provider=ModelProvider.CLAUDE,
                    tokens_used=total_tokens,
                    cost_estimate=cost,
                    latency_ms=latency_ms,
                    tool_calls=tool_calls if tool_calls else None,
                    stop_reason=response.stop_reason,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    "Claude request failed",
                    model=self.model,
                    attempt=attempt + 1,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(
                        "Retrying after backoff",
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error(
            "Claude request failed after all retries",
            model=self.model,
            max_retries=self.max_retries,
            error=str(last_error),
        )
        raise last_error

    async def stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response.

        Yields chunks of text as they arrive from Claude.

        Args:
            request: The LLM request.

        Yields:
            Text chunks from the response.
        """
        system = request.system_prompt or ""
        messages = [{"role": "user", "content": request.prompt}]

        tools_param = None
        if request.tools:
            tools_param = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system if system else None,
                messages=messages,
                tools=tools_param,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(
                "Streaming failed",
                model=self.model,
                error=str(e),
            )
            raise

    def manage_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Store messages in conversation history for context.

        Args:
            conversation_id: Unique ID for this conversation thread.
            user_message: User's message.
            assistant_response: Assistant's response.
        """
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        self.conversation_history[conversation_id].extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_response},
            ]
        )

        logger.debug(
            "Conversation stored",
            conversation_id=conversation_id,
            history_length=len(self.conversation_history[conversation_id]),
        )

    def get_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        """Retrieve conversation history.

        Args:
            conversation_id: Unique ID for the conversation.

        Returns:
            List of messages in the conversation.
        """
        return self.conversation_history.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear conversation history.

        Args:
            conversation_id: Unique ID for the conversation to clear.
        """
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
            logger.debug("Conversation cleared", conversation_id=conversation_id)

    def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Note: This is an approximation. For exact counts, use Anthropic's token counting API.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4

    # Private methods

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for a request.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        pricing = self.MODEL_PRICING.get(
            self.model,
            {"input": 3.0, "output": 15.0},  # Default to Sonnet pricing
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
