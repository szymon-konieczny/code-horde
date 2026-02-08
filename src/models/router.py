"""Multi-model router with intelligent request routing, fallback chains, and usage tracking."""

import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from enum import Enum

from pydantic import BaseModel, Field

from src.models.schemas import (
    ModelProvider,
    ModelTier,
    SensitivityLevel,
    LLMRequest,
    LLMResponse,
)


logger = structlog.get_logger(__name__)


class RoutingCondition(str, Enum):
    """Condition types for routing rules."""

    SENSITIVITY_LEVEL = "sensitivity_level"
    COMPLEXITY_SCORE = "complexity_score"
    COST_BUDGET = "cost_budget"
    PROVIDER_AVAILABILITY = "provider_availability"
    MODEL_TIER = "model_tier"


class RoutingRule(BaseModel):
    """Rule for routing requests to specific models/providers."""

    condition: RoutingCondition = Field(..., description="What condition triggers this rule")
    condition_value: Any = Field(..., description="Value to match against")
    target_provider: ModelProvider = Field(..., description="Provider to use if matched")
    target_model: Optional[str] = Field(
        default=None, description="Specific model name (optional)"
    )
    priority: int = Field(
        default=0,
        description="Higher priority rules evaluated first",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = False


@dataclass
class CircuitBreakerState:
    """Tracks circuit breaker state for a provider."""

    failure_count: int = 0
    last_failure: Optional[datetime] = None
    is_open: bool = False
    open_until: Optional[datetime] = None


@dataclass
class ProviderStats:
    """Usage statistics for a provider."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


@dataclass
class AgentUsageStats:
    """Usage statistics per agent."""

    agent_id: str
    providers: dict[ModelProvider, ProviderStats] = field(default_factory=dict)

    def get_or_create_provider_stats(self, provider: ModelProvider) -> ProviderStats:
        """Get or create stats for a provider."""
        if provider not in self.providers:
            self.providers[provider] = ProviderStats()
        return self.providers[provider]


class ModelRouter:
    """Intelligent router for distributing LLM requests across multiple models."""

    def __init__(
        self,
        routing_rules: Optional[list[RoutingRule]] = None,
        failure_threshold: int = 5,
        circuit_breaker_timeout_seconds: int = 300,
        max_queue_size: int = 100,
    ) -> None:
        """Initialize the model router.

        Args:
            routing_rules: Custom routing rules to apply.
            failure_threshold: Number of failures before opening circuit breaker.
            circuit_breaker_timeout_seconds: How long to keep circuit open.
            max_queue_size: Maximum requests to queue when both providers fail.
        """
        self.routing_rules = sorted(
            routing_rules or [], key=lambda r: r.priority, reverse=True
        )
        self.failure_threshold = failure_threshold
        self.circuit_breaker_timeout_seconds = circuit_breaker_timeout_seconds
        self.max_queue_size = max_queue_size

        # Default fallback chain — try all cloud providers, Ollama (local) last
        self.fallback_chain = [
            ModelProvider.CLAUDE,
            ModelProvider.OPENAI,
            ModelProvider.GEMINI,
            ModelProvider.KIMI,
            ModelProvider.CUSTOM,
            ModelProvider.OLLAMA,
        ]

        # Usage tracking per agent
        self.agent_stats: dict[str, AgentUsageStats] = {}

        # Request/response audit log (sanitized)
        self.audit_log: list[dict[str, Any]] = []
        self.max_audit_entries = 1000

        logger.info(
            "ModelRouter initialized",
            failure_threshold=failure_threshold,
            circuit_breaker_timeout_seconds=circuit_breaker_timeout_seconds,
        )

    def route(
        self, request: LLMRequest, agent_id: str = "unknown"
    ) -> tuple[ModelProvider, Optional[str]]:
        """Determine which provider and model should handle this request.

        Args:
            request: The LLM request.
            agent_id: ID of the agent making the request.

        Returns:
            Tuple of (provider, optional_model_name).
        """
        # Check custom routing rules first
        for rule in self.routing_rules:
            if self._matches_rule(rule, request):
                provider = rule.target_provider
                model = rule.target_model

                # Check circuit breaker
                if not self._is_provider_available(provider, agent_id):
                    logger.warning(
                        "Preferred provider unavailable, using fallback",
                        provider=provider,
                        agent_id=agent_id,
                    )
                    provider, model = self._get_fallback_provider(agent_id)
                else:
                    logger.info(
                        "Routing via custom rule",
                        rule_condition=rule.condition,
                        provider=provider,
                        agent_id=agent_id,
                    )

                return provider, model

        # Apply default routing logic
        provider, model = self._apply_default_routing(request, agent_id)

        logger.info(
            "Request routed",
            agent_id=agent_id,
            provider=provider,
            sensitivity=request.sensitivity,
            tier=request.model_preference,
        )

        return provider, model

    def record_request(
        self,
        agent_id: str,
        request: LLMRequest,
        response: LLMResponse,
        success: bool = True,
    ) -> None:
        """Record a request/response for tracking and auditing.

        Args:
            agent_id: ID of the requesting agent.
            request: The original request.
            response: The response received.
            success: Whether the request succeeded.
        """
        # Update provider stats
        agent_stats = self._get_or_create_agent_stats(agent_id)
        provider_stats = agent_stats.get_or_create_provider_stats(response.provider)

        provider_stats.total_requests += 1
        if success:
            provider_stats.successful_requests += 1
        else:
            provider_stats.failed_requests += 1
            self._record_provider_failure(response.provider, agent_id)

        if response.tokens_used:
            provider_stats.total_tokens += response.tokens_used
        if response.cost_estimate:
            provider_stats.total_cost += response.cost_estimate
        if response.latency_ms:
            provider_stats.total_latency_ms += response.latency_ms

        # Add sanitized audit entry
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "provider": response.provider.value,
            "model": response.model_used,
            "tokens": response.tokens_used,
            "cost": response.cost_estimate,
            "latency_ms": response.latency_ms,
            "success": success,
            # Prompt/response content is NOT logged for security
        }
        self.audit_log.append(audit_entry)

        # Trim audit log
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries :]

        logger.info(
            "Request recorded",
            agent_id=agent_id,
            provider=response.provider,
            success=success,
        )

    def record_failure(
        self, agent_id: str, provider: ModelProvider, error: str
    ) -> None:
        """Record a provider failure.

        Args:
            agent_id: ID of the requesting agent.
            provider: Provider that failed.
            error: Error message (sanitized).
        """
        self._record_provider_failure(provider, agent_id, error)

    def get_stats(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """Get usage statistics.

        Args:
            agent_id: Optional agent ID to get stats for. If None, returns all.

        Returns:
            Dictionary containing usage statistics.
        """
        if agent_id:
            stats = self.agent_stats.get(agent_id)
            if not stats:
                return {"agent_id": agent_id, "total_requests": 0}
            return self._format_stats(stats)

        # Aggregate stats across all agents
        total_requests = 0
        total_cost = 0.0
        total_tokens = 0
        by_provider: dict[str, dict[str, Any]] = {}

        for agent_stats in self.agent_stats.values():
            for provider, stats in agent_stats.providers.items():
                total_requests += stats.total_requests
                total_cost += stats.total_cost
                total_tokens += stats.total_tokens

                if provider.value not in by_provider:
                    by_provider[provider.value] = {
                        "requests": 0,
                        "success_rate": 0.0,
                        "cost": 0.0,
                        "tokens": 0,
                    }

                by_provider[provider.value]["requests"] += stats.total_requests
                by_provider[provider.value]["cost"] += stats.total_cost
                by_provider[provider.value]["tokens"] += stats.total_tokens

        return {
            "total_requests": total_requests,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "by_provider": by_provider,
        }

    def get_audit_log(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Get sanitized audit log entries.

        Args:
            limit: Maximum number of entries to return (most recent first).

        Returns:
            List of audit log entries.
        """
        log = list(reversed(self.audit_log))
        if limit:
            log = log[:limit]
        return log

    def reset_circuit_breaker(self, provider: ModelProvider, agent_id: str) -> None:
        """Manually reset a circuit breaker (e.g., after provider is back up).

        Args:
            provider: Provider to reset.
            agent_id: Agent ID (for tracking purposes).
        """
        agent_stats = self._get_or_create_agent_stats(agent_id)
        provider_stats = agent_stats.get_or_create_provider_stats(provider)
        provider_stats.circuit_breaker = CircuitBreakerState()
        logger.info("Circuit breaker reset", provider=provider, agent_id=agent_id)

    # Private methods

    def _matches_rule(self, rule: RoutingRule, request: LLMRequest) -> bool:
        """Check if a request matches a routing rule."""
        if rule.condition == RoutingCondition.SENSITIVITY_LEVEL:
            return request.sensitivity == rule.condition_value
        elif rule.condition == RoutingCondition.MODEL_TIER:
            return request.model_preference == rule.condition_value
        elif rule.condition == RoutingCondition.COST_BUDGET:
            # Simplified: would need to estimate cost before routing
            return True
        return False

    def _apply_default_routing(
        self, request: LLMRequest, agent_id: str
    ) -> tuple[ModelProvider, Optional[str]]:
        """Apply default routing logic based on request properties."""
        # RESTRICTED data always uses local Ollama
        if request.sensitivity == SensitivityLevel.RESTRICTED:
            return ModelProvider.OLLAMA, None

        # CONFIDENTIAL preferentially uses Claude (more secure)
        if request.sensitivity == SensitivityLevel.CONFIDENTIAL:
            if self._is_provider_available(ModelProvider.CLAUDE, agent_id):
                return ModelProvider.CLAUDE, None

        # POWERFUL tier uses Claude
        if request.model_preference == ModelTier.POWERFUL:
            if self._is_provider_available(ModelProvider.CLAUDE, agent_id):
                return ModelProvider.CLAUDE, None

        # Default: try Claude first (best quality), fallback to Ollama
        for provider in self.fallback_chain:
            if self._is_provider_available(provider, agent_id):
                return provider, None

        # All providers down - return first in fallback chain
        return self.fallback_chain[0], None

    def _is_provider_available(self, provider: ModelProvider, agent_id: str) -> bool:
        """Check if a provider's circuit breaker is not open."""
        agent_stats = self._get_or_create_agent_stats(agent_id)
        provider_stats = agent_stats.get_or_create_provider_stats(provider)
        cb = provider_stats.circuit_breaker

        if not cb.is_open:
            return True

        # Check if timeout has elapsed
        if cb.open_until and datetime.utcnow() >= cb.open_until:
            cb.is_open = False
            cb.failure_count = 0
            cb.last_failure = None
            logger.info(
                "Circuit breaker closed",
                provider=provider,
                agent_id=agent_id,
            )
            return True

        return False

    def _get_fallback_provider(
        self, agent_id: str
    ) -> tuple[ModelProvider, Optional[str]]:
        """Get the next available provider in fallback chain."""
        for provider in self.fallback_chain:
            if self._is_provider_available(provider, agent_id):
                logger.info(
                    "Using fallback provider",
                    provider=provider,
                    agent_id=agent_id,
                )
                return provider, None

        logger.warning(
            "No providers available, using fallback chain head",
            agent_id=agent_id,
        )
        return self.fallback_chain[0], None

    def _record_provider_failure(
        self, provider: ModelProvider, agent_id: str, error: str = ""
    ) -> None:
        """Record a failure and update circuit breaker."""
        agent_stats = self._get_or_create_agent_stats(agent_id)
        provider_stats = agent_stats.get_or_create_provider_stats(provider)
        cb = provider_stats.circuit_breaker

        cb.failure_count += 1
        cb.last_failure = datetime.utcnow()

        logger.warning(
            "Provider failure recorded",
            provider=provider,
            agent_id=agent_id,
            failure_count=cb.failure_count,
            threshold=self.failure_threshold,
            error=error[:100],  # Truncate for logging
        )

        if cb.failure_count >= self.failure_threshold:
            cb.is_open = True
            cb.open_until = datetime.utcnow() + timedelta(
                seconds=self.circuit_breaker_timeout_seconds
            )
            logger.error(
                "Circuit breaker opened",
                provider=provider,
                agent_id=agent_id,
                open_until=cb.open_until.isoformat(),
            )

    def _get_or_create_agent_stats(self, agent_id: str) -> AgentUsageStats:
        """Get or create usage stats for an agent."""
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = AgentUsageStats(agent_id=agent_id)
        return self.agent_stats[agent_id]

    def _format_stats(self, agent_stats: AgentUsageStats) -> dict[str, Any]:
        """Format agent stats for output."""
        result = {
            "agent_id": agent_stats.agent_id,
            "total_requests": sum(s.total_requests for s in agent_stats.providers.values()),
            "total_cost": sum(s.total_cost for s in agent_stats.providers.values()),
            "total_tokens": sum(s.total_tokens for s in agent_stats.providers.values()),
            "by_provider": {},
        }

        for provider, stats in agent_stats.providers.items():
            result["by_provider"][provider.value] = {
                "requests": stats.total_requests,
                "successful": stats.successful_requests,
                "failed": stats.failed_requests,
                "success_rate_percent": stats.success_rate,
                "cost": stats.total_cost,
                "tokens": stats.total_tokens,
                "avg_latency_ms": stats.average_latency_ms,
            }

        return result


# Alias so main.py can do: from src.models.router import LLMRouter
LLMRouter = ModelRouter
