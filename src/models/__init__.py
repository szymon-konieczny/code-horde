"""Models package for LLM integration and routing."""

from src.models.schemas import (
    ModelProvider,
    ModelTier,
    SensitivityLevel,
    LLMRequest,
    LLMResponse,
    ToolDefinition,
    ToolCall,
)
from src.models.router import ModelRouter, LLMRouter, RoutingRule
from src.models.router_extensions import (
    RouterExtensions,
    EnhancedModelRouter,
    extend_router_fallback_chain,
    add_openai_routing_rules,
    add_huggingface_routing_rules,
    add_bielik_routing_rules,
)
from src.models.claude_client import ClaudeClient
from src.models.openai_client import OpenAIClient
from src.models.gemini_client import GeminiClient
from src.models.kimi_client import KimiClient
from src.models.huggingface_client import HuggingFaceClient
from src.models.bielik_client import BielikClient
from src.models.ollama_client import OllamaClient
from src.models.custom_openai_client import CustomOpenAIClient
from src.models.rlm_engine import (
    RLMEngine,
    RLMConfig,
    RLMQuery,
    ContextRegistry,
    ContextPartition,
)

__all__ = [
    # Core schemas
    "ModelProvider",
    "ModelTier",
    "SensitivityLevel",
    "LLMRequest",
    "LLMResponse",
    "ToolDefinition",
    "ToolCall",
    # Routing
    "ModelRouter",
    "LLMRouter",
    "EnhancedModelRouter",
    "RoutingRule",
    "RouterExtensions",
    "extend_router_fallback_chain",
    "add_openai_routing_rules",
    "add_huggingface_routing_rules",
    "add_bielik_routing_rules",
    # Client implementations
    "ClaudeClient",
    "OpenAIClient",
    "GeminiClient",
    "KimiClient",
    "HuggingFaceClient",
    "BielikClient",
    "OllamaClient",
    "CustomOpenAIClient",
    # RLM engine
    "RLMEngine",
    "RLMConfig",
    "RLMQuery",
    "ContextRegistry",
    "ContextPartition",
]
