"""Shared Pydantic models for LLM requests, responses, and tools."""

from enum import Enum
from typing import Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    KIMI = "kimi"
    HUGGINGFACE = "huggingface"
    BIELIK = "bielik"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class ModelTier(str, Enum):
    """Model performance/capability tiers."""

    FAST = "fast"
    BALANCED = "balanced"
    POWERFUL = "powerful"


class SensitivityLevel(str, Enum):
    """Data sensitivity classification for routing decisions."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ToolDefinition(BaseModel):
    """Schema for a tool/function that an LLM can use."""

    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(
        ..., description="Human-readable description of what the tool does"
    )
    input_schema: dict[str, Any] = Field(
        ..., description="JSON schema describing the tool's input parameters"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "name": "send_message",
                "description": "Send a message to another agent",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Target agent ID"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["agent_id", "message"],
                },
            }
        }


class ToolCall(BaseModel):
    """Represents a tool invocation made by the LLM."""

    tool_name: str = Field(..., description="Name of the tool being called")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    id: Optional[str] = Field(
        default=None, description="Unique ID for this tool call (from provider)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "tool_name": "send_message",
                "id": "toolu_01ABC123XYZ",
                "arguments": {"agent_id": "agent-42", "message": "Status update"},
            }
        }


class LLMRequest(BaseModel):
    """Request schema for LLM inference."""

    prompt: str = Field(..., description="The main user prompt")
    system_prompt: Optional[str] = Field(
        default=None, description="System message to guide LLM behavior"
    )
    model_preference: Optional[ModelTier] = Field(
        default=ModelTier.BALANCED,
        description="Preferred model tier (FAST, BALANCED, POWERFUL)",
    )
    sensitivity: Optional[SensitivityLevel] = Field(
        default=SensitivityLevel.INTERNAL,
        description="Data sensitivity level for routing decisions",
    )
    max_tokens: Optional[int] = Field(
        default=2048, description="Maximum tokens in response"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)",
    )
    tools: Optional[list[ToolDefinition]] = Field(
        default=None, description="List of tools the LLM can call"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "prompt": "What are the top 3 issues in the backlog?",
                "system_prompt": "You are a helpful project manager assistant.",
                "model_preference": "balanced",
                "sensitivity": "internal",
                "max_tokens": 1024,
                "temperature": 0.7,
            }
        }


class LLMResponse(BaseModel):
    """Response schema from LLM inference."""

    content: str = Field(..., description="The LLM's response content")
    model_used: str = Field(..., description="Name of the model that generated this")
    provider: ModelProvider = Field(..., description="Provider that served this request")
    tokens_used: Optional[int] = Field(
        default=None,
        description="Total tokens consumed (input + output)",
    )
    cost_estimate: Optional[float] = Field(
        default=None,
        description="Estimated cost in USD",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Response time in milliseconds",
    )
    tool_calls: Optional[list[ToolCall]] = Field(
        default=None, description="Tool calls made by the LLM"
    )
    stop_reason: Optional[str] = Field(
        default=None,
        description="Reason response was generated (e.g., 'end_turn', 'tool_use')",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "content": "The top 3 issues are: 1) Auth bug, 2) API timeout, 3) UI crash",
                "model_used": "claude-3-sonnet-20240229",
                "provider": "claude",
                "tokens_used": 256,
                "cost_estimate": 0.00089,
                "latency_ms": 450.5,
                "stop_reason": "end_turn",
            }
        }
