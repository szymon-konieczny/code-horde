"""Base agent class and types for AgentArmy multi-agent system."""

import hashlib
import hmac
import os
import pathlib
import time
import yaml
import traceback as _traceback_mod
import uuid
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.reasoning import ChainOfThoughtEngine, ReasoningChain, ReasoningStrategy

logger = structlog.get_logger(__name__)


class AgentState(str, Enum):
    """Agent operational states.

    Attributes:
        IDLE: Agent is ready to accept tasks.
        BUSY: Agent is processing a task.
        PAUSED: Agent is temporarily paused.
        ERROR: Agent encountered an error.
        OFFLINE: Agent is offline/unreachable.
    """

    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"


class AgentCapability(BaseModel):
    """Represents a capability an agent possesses.

    Attributes:
        name: Name of the capability.
        version: Capability version.
        parameters: Parameters accepted by this capability.
        description: Human-readable description.
    """

    name: str = Field(description="Capability name")
    version: str = Field(default="1.0.0", description="Capability version")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Capability parameters"
    )
    description: str = Field(default="", description="Capability description")


class AgentIdentity(BaseModel):
    """Agent identity and metadata.

    Attributes:
        id: Unique agent identifier.
        name: Agent display name.
        role: Agent role/type.
        capabilities: List of agent capabilities.
        security_level: Security clearance level (1-5, where 5 is highest).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique agent ID")
    name: str = Field(description="Agent name")
    role: str = Field(description="Agent role")
    capabilities: list[AgentCapability] = Field(
        default_factory=list, description="Available capabilities"
    )
    security_level: int = Field(default=1, description="Security level (1-5)", ge=1, le=5)


class AgentHeartbeat(BaseModel):
    """Agent heartbeat status report.

    Attributes:
        agent_id: ID of the reporting agent.
        timestamp: Time of the heartbeat.
        state: Current agent state.
        uptime_seconds: Uptime in seconds.
        tasks_completed: Number of completed tasks.
        tasks_failed: Number of failed tasks.
    """

    agent_id: str = Field(description="Agent ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: AgentState = Field(description="Current state")
    uptime_seconds: float = Field(ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)


class BaseAgent(ABC):
    """Abstract base class for all agents in AgentArmy.

    Provides lifecycle management, capability checking, audit logging, and
    integration with the LLM router.

    Attributes:
        identity: Agent identity and metadata.
        startup_time: When the agent started.
    """

    def __init__(self, identity: AgentIdentity) -> None:
        """Initialize a new agent.

        Args:
            identity: Agent identity and metadata.
        """
        self.identity = identity
        self.startup_time = time.time()
        self._state = AgentState.OFFLINE
        self._task_count_completed = 0
        self._task_count_failed = 0
        self._logger = structlog.get_logger(self.identity.name)
        # Error ring buffer — stores last N errors for API/dashboard inspection
        self._error_log: deque[dict[str, Any]] = deque(maxlen=100)
        # References for inter-agent task creation — set by the orchestrator
        self._task_manager: Any = None
        self._orchestrator: Any = None

    @property
    def state(self) -> AgentState:
        """Get current agent state.

        Returns:
            Current operational state.
        """
        return self._state

    @property
    def uptime_seconds(self) -> float:
        """Get agent uptime in seconds.

        Returns:
            Seconds since agent startup.
        """
        return time.time() - self.startup_time

    def _log_error(
        self,
        task_id: str,
        error: Exception,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append an error to the agent's in-memory ring buffer.

        Args:
            task_id: ID of the task that caused the error.
            error: The exception that was raised.
            context: Optional extra context (task type, payload snippet, etc.).
        """
        self._error_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "traceback": _traceback_mod.format_exc(),
            "context": context or {},
        })

    def get_error_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent errors, newest first.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of error dicts ordered newest-first.
        """
        entries = list(self._error_log)
        return list(reversed(entries[-limit:]))

    def get_system_context_public(self) -> str:
        """Return the agent's system context string (public wrapper).

        Returns:
            The system context used in LLM prompts.
        """
        return self._get_system_context()

    async def startup(self) -> None:
        """Initialize and start the agent.

        Called once during agent lifecycle initialization. Should establish
        connections, load models, and prepare resources.

        Raises:
            Exception: If startup fails.
        """
        try:
            self._state = AgentState.IDLE
            await self._logger.ainfo(
                "agent_startup_success",
                agent_id=self.identity.id,
                agent_name=self.identity.name,
            )
        except Exception as exc:
            self._state = AgentState.ERROR
            await self._logger.aerror(
                "agent_startup_failed",
                agent_id=self.identity.id,
                error=str(exc),
            )
            raise

    async def shutdown(self) -> None:
        """Shutdown the agent gracefully.

        Called during agent lifecycle termination. Should cleanup resources,
        close connections, and persist state.

        Raises:
            Exception: If shutdown fails.
        """
        try:
            self._state = AgentState.OFFLINE
            await self._logger.ainfo(
                "agent_shutdown_success",
                agent_id=self.identity.id,
                uptime_seconds=self.uptime_seconds,
            )
        except Exception as exc:
            await self._logger.aerror(
                "agent_shutdown_failed",
                agent_id=self.identity.id,
                error=str(exc),
            )
            raise

    def _get_system_context(self) -> str:
        """Return agent-specific system context for LLM reasoning.

        Override in subclasses to provide role-specific context.

        Returns:
            System context string describing the agent's expertise.
        """
        caps = ", ".join(cap.name for cap in self.identity.capabilities)
        return (
            f"Your capabilities include: {caps}. "
            f"Security level: {self.identity.security_level}/5."
        )

    def _load_custom_instructions(self) -> dict[str, str]:
        """Load custom prepend/append instructions from agent config YAML.

        Reads ``config/agents/{prefix}.yaml`` and extracts the
        ``custom_instructions`` section.  Returns empty strings on any
        failure so the agent always works even without a config file.

        Returns:
            Dict with ``prepend`` and ``append`` keys.
        """
        prefix = self.identity.id.split("-")[0] if "-" in self.identity.id else self.identity.id
        config_path = pathlib.Path("config/agents") / f"{prefix}.yaml"

        result: dict[str, str] = {"prepend": "", "append": ""}

        if not config_path.exists():
            return result

        try:
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_data and isinstance(config_data, dict):
                ci = config_data.get("custom_instructions", {})
                if isinstance(ci, dict):
                    result["prepend"] = str(ci.get("prepend", "") or "").strip()
                    result["append"] = str(ci.get("append", "") or "").strip()
        except Exception:
            pass  # Non-critical — agent works without custom instructions

        return result

    async def think(
        self,
        task: dict[str, Any],
        strategy: Optional[str] = None,
    ) -> "ReasoningChain":
        """Apply chain-of-thought reasoning to a task using an LLM.

        Builds a reasoning prompt, calls the LLM via the available client,
        and parses the response into a structured ReasoningChain.

        Args:
            task: Task payload to reason about.
            strategy: Reasoning strategy name (step_by_step, react,
                tree_of_thought, self_critique). Defaults to step_by_step.

        Returns:
            ReasoningChain with structured thinking steps and conclusion.
        """
        from src.core.reasoning import (
            ChainOfThoughtEngine,
            ReasoningStrategy,
        )
        from src.models.schemas import (
            LLMRequest,
            LLMResponse,
            ModelProvider,
            ModelTier,
            SensitivityLevel,
        )

        engine = ChainOfThoughtEngine()

        # Resolve strategy
        if strategy is None:
            reasoning_strategy = ReasoningStrategy.STEP_BY_STEP
        else:
            reasoning_strategy = ReasoningStrategy(strategy)

        # Build the CoT-enriched LLM request (sync method — no await)
        llm_request = engine.build_reasoning_request(
            task=task,
            agent_identity=self.identity,
            system_context=self._get_system_context(),
            strategy=reasoning_strategy,
        )

        # Try each configured LLM provider in order
        start_ms = time.time() * 1000
        llm_response = None

        for provider_name, api_key, model in self._get_llm_providers():
            try:
                client = self._create_llm_client(provider_name, api_key, model_override=model)
                llm_response = await client.complete(llm_request)
                break
            except Exception as exc:
                await self._logger.adebug(
                    "cot_provider_failed",
                    agent_id=self.identity.id,
                    provider=provider_name,
                    error=str(exc)[:120],
                )
                continue

        if llm_response is None:
            # All providers exhausted — use a synthetic reasoning response
            await self._logger.awarning(
                "cot_all_providers_failed",
                agent_id=self.identity.id,
            )
            elapsed = time.time() * 1000 - start_ms
            llm_response = LLMResponse(
                content=(
                    f'<step title="Analysis">\n'
                    f'Analyzing task: {task.get("description", task.get("type", "unknown"))}. '
                    f"Context: {task.get('context', {})}.\n"
                    f"</step>\n\n"
                    f"<conclusion>\n"
                    f"Proceeding with standard {self.identity.role} workflow.\n"
                    f"</conclusion>"
                ),
                model_used="fallback",
                provider=ModelProvider.OLLAMA,
                tokens_used=0,
                latency_ms=elapsed,
            )

        # Parse the response into a structured reasoning chain
        chain = engine.parse_reasoning_response(
            response=llm_response,
            task=task,
            agent_identity=self.identity,
            strategy=reasoning_strategy,
        )

        await self._logger.ainfo(
            "cot_reasoning_complete",
            agent_id=self.identity.id,
            strategy=reasoning_strategy.value,
            steps=len(chain.steps),
            conclusion_length=len(chain.conclusion),
            total_tokens=chain.total_tokens,
        )

        return chain

    def _get_llm_providers(self) -> list[tuple[str, Optional[str], Optional[str]]]:
        """Build an ordered list of (provider_name, api_key, model) to try.

        Returns providers in preference order: Claude → OpenAI → Gemini →
        Kimi → Ollama (local, always last).  Only includes providers whose
        API key is actually set.  The model comes from env config.
        """
        import os

        providers: list[tuple[str, Optional[str], Optional[str]]] = []

        for name, env_var in [
            ("claude", "AGENTARMY_CLAUDE_API_KEY"),
            ("openai", "AGENTARMY_OPENAI_API_KEY"),
            ("gemini", "AGENTARMY_GEMINI_API_KEY"),
            ("kimi", "AGENTARMY_KIMI_API_KEY"),
            ("custom", "AGENTARMY_CUSTOM_API_KEY"),
        ]:
            key = os.environ.get(env_var, "")
            if key and not key.startswith(("your_", "YOUR_")):
                model = self._get_llm_model(name)
                providers.append((name, key, model))

        # Ollama is always the last resort (local, no key needed)
        providers.append(("ollama", None, self._get_llm_model("ollama")))
        return providers

    def _resolve_model_override(
        self,
        model_id: str,
        providers: list[tuple[str, Optional[str], Optional[str]]],
    ) -> list[tuple[str, Optional[str], Optional[str]]]:
        """Reorder providers so the user-chosen model is tried first.

        Maps model IDs to providers and inserts a targeted entry at the front.

        Args:
            model_id: Model ID from the dashboard selector (e.g. "claude-sonnet-4-20250514").
            providers: Existing ordered provider list.

        Returns:
            Updated list with the override prepended.
        """
        import os

        # Map model IDs to their provider
        MODEL_TO_PROVIDER = {
            # Claude
            "claude-opus-4-6": "claude",
            "claude-sonnet-4-5-20250929": "claude",
            "claude-haiku-4-5-20251001": "claude",
            "claude-sonnet-4-20250514": "claude",
            "claude-opus-4-20250514": "claude",
            "claude-3-5-sonnet-20241022": "claude",
            # OpenAI
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "o1": "openai",
            "o3-mini": "openai",
            # Google
            "gemini-2.0-flash": "gemini",
            "gemini-2.0-flash-lite": "gemini",
            "gemini-2.5-pro": "gemini",
            # Kimi
            "kimi-k2.5": "kimi",
            "moonshot-v1-8k": "kimi",
            "moonshot-v1-32k": "kimi",
            "moonshot-v1-128k": "kimi",
            # Custom (OpenAI-compatible)
            "custom": "custom",
            # Local
            "ollama": "ollama",
        }

        provider_name = MODEL_TO_PROVIDER.get(model_id)
        if not provider_name:
            return providers  # Unknown model, use default chain

        # Get API key for this provider
        env_map = {
            "claude": "AGENTARMY_CLAUDE_API_KEY",
            "openai": "AGENTARMY_OPENAI_API_KEY",
            "gemini": "AGENTARMY_GEMINI_API_KEY",
            "kimi": "AGENTARMY_KIMI_API_KEY",
            "custom": "AGENTARMY_CUSTOM_API_KEY",
        }
        api_key = os.environ.get(env_map.get(provider_name, ""), "") or None
        if provider_name == "ollama":
            api_key = None

        # For "ollama", don't set a model override — use the env-configured one
        model_for_override = model_id if model_id != "ollama" else self._get_llm_model("ollama")

        # Prepend the override, then keep remaining providers as fallbacks
        result = [(provider_name, api_key, model_for_override)]
        for p in providers:
            if p[0] != provider_name:  # avoid duplicate
                result.append(p)
        return result

    def _get_llm_model(self, provider_name: str) -> Optional[str]:
        """Get the configured model for a provider from settings.

        Args:
            provider_name: Provider name (claude, openai, gemini, kimi).

        Returns:
            Configured model string, or None to use client default.
        """
        import os
        model_env = {
            "claude": "AGENTARMY_CLAUDE_DEFAULT_MODEL",
            "openai": "AGENTARMY_OPENAI_DEFAULT_MODEL",
            "gemini": "AGENTARMY_GEMINI_DEFAULT_MODEL",
            "kimi": "AGENTARMY_KIMI_DEFAULT_MODEL",
            "custom": "AGENTARMY_CUSTOM_DEFAULT_MODEL",
            "ollama": "AGENTARMY_OLLAMA_DEFAULT_MODEL",
        }
        env_var = model_env.get(provider_name)
        if env_var:
            return os.environ.get(env_var) or None
        return None

    def _create_llm_client(self, provider_name: str, api_key: Optional[str], model_override: Optional[str] = None) -> Any:
        """Create an LLM client for the given provider.

        Args:
            provider_name: One of claude, openai, gemini, kimi, ollama.
            api_key: API key (None for Ollama).
            model_override: Specific model to use (takes precedence over env config).

        Returns:
            Client instance with a .complete() method.
        """
        model = model_override or self._get_llm_model(provider_name)

        if provider_name == "claude":
            from src.models.claude_client import ClaudeClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return ClaudeClient(**kwargs)
        elif provider_name == "openai":
            from src.models.openai_client import OpenAIClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return OpenAIClient(**kwargs)
        elif provider_name == "gemini":
            from src.models.gemini_client import GeminiClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return GeminiClient(**kwargs)
        elif provider_name == "kimi":
            from src.models.kimi_client import KimiClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return KimiClient(**kwargs)
        elif provider_name == "custom":
            from src.models.custom_openai_client import CustomOpenAIClient
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return CustomOpenAIClient(**kwargs)
        elif provider_name == "ollama":
            from src.models.ollama_client import OllamaClient
            kwargs = {}
            if model:
                kwargs["model"] = model
            return OllamaClient(**kwargs)
        raise ValueError(f"Unknown provider: {provider_name}")

    # ── Project context gathering ─────────────────────────────────

    # Files that commonly contain project-level context.  Listed in
    # priority order so the most important files are read first.
    # Agent instruction files — loaded first and injected as top-priority context
    _AGENT_INSTRUCTION_FILES = [
        "AGENTS.md", "AGENTS.txt",        # Primary agent instruction file
        ".agents.md", ".agents",           # Dot-prefixed variant
        "AI_RULES.md", "AI.md",            # Alternative naming
        "CLAUDE.md",                       # Claude-specific rules
        ".cursorrules",                    # Cursor AI rules (compatible)
        ".github/copilot-instructions.md", # GitHub Copilot instructions
    ]

    _PROJECT_ROOT_FILES = [
        "README.md", "README.rst", "README.txt", "README",
        "DEPLOYMENT.md", "DEPLOY.md", "DEPLOYING.md",
        "CONTRIBUTING.md", "ARCHITECTURE.md",
        "Makefile", "Justfile",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "Procfile", "fly.toml", "render.yaml", "vercel.json",
        "netlify.toml", "app.yaml", "serverless.yml",
        "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
    ]

    # Directories to scan one level deep for extra docs.
    _DOC_DIRS = ["docs", "doc", ".github", "deploy", "deployment", "infra", "infrastructure", "scripts"]

    # File extensions we consider readable documentation / config.
    _READABLE_EXTS = {
        ".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".json",
        ".cfg", ".ini", ".env.example", ".sh", ".bash", ".tf",
    }

    _MAX_FILE_SIZE = 8_000   # chars per file (truncate beyond this)
    _MAX_TOTAL_CTX = 24_000  # total chars budget for project context

    async def _load_agent_instructions(self, project_dir: str) -> str:
        """Load AGENTS.md (or equivalent) from the project root.

        This file contains project-specific rules, conventions, and
        coding guidelines that should take top priority in agent behavior.

        Returns:
            Formatted instruction string, or empty string if not found.
        """
        root = pathlib.Path(project_dir)
        if not root.is_dir():
            return ""

        for name in self._AGENT_INSTRUCTION_FILES:
            fpath = root / name
            if fpath.is_file():
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    if not content.strip():
                        continue
                    # Truncate very large instruction files
                    if len(content) > 12_000:
                        content = content[:12_000] + "\n... (truncated)"
                    return (
                        f"\n\n=== PROJECT AGENT INSTRUCTIONS ({name}) ===\n"
                        f"THE FOLLOWING RULES ARE FROM THE PROJECT'S OWN INSTRUCTION FILE.\n"
                        f"THESE TAKE PRECEDENCE OVER GENERAL BEHAVIOR.\n\n"
                        f"{content}\n"
                        f"=== END PROJECT AGENT INSTRUCTIONS ===\n"
                    )
                except (IOError, PermissionError):
                    continue

        return ""

    async def _gather_project_context(
        self,
        project_dir: str,
        query: str,
    ) -> str:
        """Scan the project directory and gather relevant file contents.

        Reads README, deployment docs, configs, and CI files from the
        project root and common documentation directories.  Returns a
        formatted string with file contents ready to be injected into
        the LLM system prompt.

        Args:
            project_dir: Absolute path to the project root.
            query: The user's message, used to prioritise relevant files.

        Returns:
            A formatted context string with file contents, or empty string.
        """
        root = pathlib.Path(project_dir)
        if not root.is_dir():
            return ""

        query_lower = query.lower()

        # ── Collect candidate files ──────────────────────────────
        candidates: list[pathlib.Path] = []

        # 1) Root-level important files
        for name in self._PROJECT_ROOT_FILES:
            p = root / name
            if p.is_file():
                candidates.append(p)

        # 2) Files inside documentation directories (one level deep)
        for dname in self._DOC_DIRS:
            dpath = root / dname
            if dpath.is_dir():
                try:
                    for entry in sorted(dpath.iterdir()):
                        if entry.is_file() and entry.suffix.lower() in self._READABLE_EXTS:
                            candidates.append(entry)
                except PermissionError:
                    continue

        # 3) CI config files (nested paths)
        for ci_path in [
            ".github/workflows", ".gitlab-ci.yml", ".circleci/config.yml",
            "Jenkinsfile", "bitbucket-pipelines.yml",
        ]:
            p = root / ci_path
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                try:
                    for entry in sorted(p.iterdir()):
                        if entry.is_file() and entry.suffix.lower() in {".yml", ".yaml"}:
                            candidates.append(entry)
                except PermissionError:
                    continue

        if not candidates:
            return ""

        # ── Deduplicate ──────────────────────────────────────────
        seen: set[str] = set()
        unique: list[pathlib.Path] = []
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # ── Score and sort by relevance to the query ─────────────
        deploy_keywords = {"deploy", "deployment", "prod", "production", "release", "ci", "cd", "pipeline", "docker", "infrastructure", "hosting"}
        build_keywords = {"build", "compile", "install", "setup", "make", "run", "start", "dev"}
        test_keywords = {"test", "testing", "qa", "quality", "coverage", "lint"}
        security_keywords = {"security", "auth", "ssl", "tls", "secret", "credential", "vulnerability"}

        def relevance_score(p: pathlib.Path) -> int:
            """Higher score = more relevant to the query."""
            name_lower = p.name.lower()
            stem = p.stem.lower()
            content_hint = name_lower + " " + str(p.relative_to(root)).lower()
            score = 0

            # README always gets a base score
            if name_lower.startswith("readme"):
                score += 5

            # Match file names against query keywords
            query_words = set(query_lower.split())
            for word in query_words:
                if word in content_hint:
                    score += 10

            # Boost deploy-related files for deploy queries
            if query_words & deploy_keywords:
                if any(kw in content_hint for kw in deploy_keywords):
                    score += 15
            if query_words & build_keywords:
                if any(kw in content_hint for kw in build_keywords):
                    score += 15
            if query_words & test_keywords:
                if any(kw in content_hint for kw in test_keywords):
                    score += 15
            if query_words & security_keywords:
                if any(kw in content_hint for kw in security_keywords):
                    score += 15

            return score

        unique.sort(key=lambda p: relevance_score(p), reverse=True)

        # ── Read files up to budget ──────────────────────────────
        sections: list[str] = []
        total_chars = 0

        for fpath in unique:
            if total_chars >= self._MAX_TOTAL_CTX:
                break
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except (IOError, PermissionError):
                continue

            # Skip binary / very large files
            if "\x00" in content[:200]:
                continue

            # Truncate individual files
            if len(content) > self._MAX_FILE_SIZE:
                content = content[: self._MAX_FILE_SIZE] + "\n... (truncated)"

            rel_path = fpath.relative_to(root)
            section = f"── {rel_path} ──\n{content}"
            sections.append(section)
            total_chars += len(section)

        if not sections:
            return ""

        # ── Build the directory tree (shallow) for orientation ───
        tree_lines: list[str] = []
        try:
            for entry in sorted(root.iterdir()):
                if entry.name.startswith(".") and entry.name not in {".github", ".gitlab-ci.yml"}:
                    continue
                marker = "/" if entry.is_dir() else ""
                tree_lines.append(f"  {entry.name}{marker}")
        except PermissionError:
            pass

        tree_str = "\n".join(tree_lines[:40])  # cap at 40 entries
        if len(tree_lines) > 40:
            tree_str += f"\n  ... ({len(tree_lines) - 40} more entries)"

        return (
            f"\n\n=== PROJECT CONTEXT ===\n"
            f"Project root: {project_dir}\n"
            f"Directory structure:\n{tree_str}\n\n"
            f"Relevant project files:\n\n"
            + "\n\n".join(sections)
            + "\n=== END PROJECT CONTEXT ===\n"
        )

    async def _handle_chat_message(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle a free-form chat message using LLM.

        Tries each configured LLM provider in order (Claude → OpenAI →
        Gemini → Kimi → Ollama).  If task payload includes a 'model' key,
        that specific provider+model is tried first.

        Falls back to a friendly template when no provider is reachable.

        Args:
            task: Task dict with 'description' containing the user message.

        Returns:
            Dict with 'response' key containing the agent's reply.
        """
        message = task.get("description", "")
        task_payload = task.get("payload", {}) or {}
        model_override = task_payload.get("model")  # e.g. "claude-sonnet-4-20250514" or "ollama"

        from src.models.schemas import (
            LLMRequest,
            ModelTier,
            SensitivityLevel,
        )

        caps = ", ".join(cap.name for cap in self.identity.capabilities)
        project_dir = task_payload.get("project_dir", "")

        # Load AGENTS.md / project instruction file (highest priority)
        agent_instructions = ""
        if project_dir:
            try:
                agent_instructions = await self._load_agent_instructions(project_dir)
            except Exception:
                pass  # Non-critical

        # Gather actual file contents from the project directory
        project_context = ""
        if project_dir:
            try:
                project_context = await self._gather_project_context(project_dir, message)
            except Exception as ctx_exc:
                await self._logger.awarning(
                    "project_context_gathering_failed",
                    agent_id=self.identity.id,
                    error=str(ctx_exc)[:200],
                )
                project_context = f"\nProject directory: {project_dir}\n"

        # Gather URL context (Jira issues, Figma designs) if resolved
        url_context = task_payload.get("url_context", "")
        if url_context:
            url_context = f"\n\n--- LINKED RESOURCES ---\n{url_context}\n--- END LINKED RESOURCES ---\n"

        # Gather cross-session memory context
        memory_context = task_payload.get("memory_context", "")
        if memory_context:
            memory_context = f"\n\n{memory_context}\n"

        # Gather configured integrations context
        integrations_context = task_payload.get("integrations_context", "")
        if integrations_context:
            integrations_context = f"\n\n{integrations_context}\n"

        # Load user-customisable instructions from config YAML
        _ci = self._load_custom_instructions()
        _prepend = (
            f"\n\n=== CUSTOM INSTRUCTIONS (PREPEND) ===\n{_ci['prepend']}\n=== END CUSTOM INSTRUCTIONS ===\n"
            if _ci["prepend"] else ""
        )
        _append = (
            f"\n\n=== CUSTOM INSTRUCTIONS (APPEND) ===\n{_ci['append']}\n=== END CUSTOM INSTRUCTIONS ===\n"
            if _ci["append"] else ""
        )

        system_prompt = (
            f"You are {self.identity.name}, a specialized AI agent in the AgentArmy system. "
            f"Your role is: {self.identity.role}. "
            f"{self._get_system_context()}"
            f"{_prepend}\n\n"
            f"Your capabilities: {caps}.\n\n"
            f"## TOOL ACCESS\n"
            f"You have access to a terminal/shell in the project directory. "
            f"You can execute git commands, build tools, linters, test runners, package managers, and any CLI tool.\n\n"
            f"**CRITICAL: When to show vs. hide bash commands:**\n"
            f"- **Show as ```bash blocks** ONLY for commands the user needs to run: installs, builds, deploys, git operations, fixes. "
            f"These get a Run button in the dashboard.\n"
            f"- **DO NOT show bash blocks** for your own exploration/analysis commands like `find`, `cat`, `grep`, `ls`, `head`, `wc`. "
            f"Instead, describe what you found in prose. The user wants **results**, not a list of commands you would run.\n"
            f"- When analyzing a codebase, reading files, or exploring project structure: "
            f"present your findings as organized prose/lists — NOT as bash commands the user has to run one by one.\n"
            f"- Example of WRONG output: showing 7 bash blocks like `find src -name '*.controller.ts'`, `grep -rn '@Get'`, `cat src/main.ts`\n"
            f"- Example of RIGHT output: 'The project has 5 controllers with 23 endpoints: ...' (organized summary)\n\n"
            f"Integration tokens (Figma, GitHub, Jira) are injected as environment variables when commands run. "
            f"Use `$FIGMA_TOKEN`, `$GITHUB_TOKEN`, `$JIRA_API_TOKEN`, `$JIRA_EMAIL`, `$JIRA_DOMAIN` in bash blocks. "
            f"NEVER use placeholder tokens like `<YOUR_FIGMA_TOKEN>` or `<YOUR_TOKEN>` — real tokens are already available.\n"
            f"{integrations_context}\n"
            f"{agent_instructions}\n"
            f"{memory_context}\n"
            f"{project_context}\n"
            f"{url_context}\n"
            f"## REASONING APPROACH\n"
            f"Before answering, think through the problem systematically:\n"
            f"1. **Understand** — Identify what the user is really asking and the core requirements\n"
            f"2. **Analyze** — Consider your domain expertise, the project context, and any constraints\n"
            f"3. **Plan** — Decide on your approach before executing\n"
            f"4. **Execute** — Carry out the plan with attention to detail\n"
            f"5. **Verify** — Check your reasoning and output for correctness\n"
            f"For complex questions, structure your response with clear sections. "
            f"Show your reasoning when it helps the user understand.\n\n"
            f"Respond helpfully and concisely to the user's message. "
            f"When project agent instructions are provided (AGENTS.md), ALWAYS follow those rules. "
            f"When project files are provided, always reference specific details from them. "
            f"Prefer project-specific information over generic advice. "
            f"When Jira issues or Figma designs are linked, incorporate their details into your response. "
            f"Reference specific issue fields, design components, or frame names as relevant. "
            f"If the answer is in the project docs, quote or reference the relevant file. "
            f"Present your analysis and findings as clear, organized prose. "
            f"Only use ```bash blocks for commands the user actually needs to execute (installs, builds, deploys, fixes). "
            f"Keep responses clear and actionable.\n\n"
            f"## OUTPUT FORMATTING\n"
            f"When presenting design data (Figma tokens, colors, typography, etc.), "
            f"ALWAYS format results as a clean, human-readable summary organized by category:\n"
            f"- **Colors**: Show hex values with names (e.g., `Primary: #FF5733`, `Text Dark: #333333`)\n"
            f"- **Typography**: Show font specs (e.g., `Heading 1: Inter, 32px, Bold, 1.2 line-height`)\n"
            f"- **Spacing**: List values (e.g., `8px, 16px, 24px, 32px`)\n"
            f"- **Shadows**: Show CSS (e.g., `Card Shadow: 0 2px 4px rgba(0,0,0,0.1)`)\n"
            f"- **Border Radii**: List values (e.g., `4px, 8px, 12px, 24px`)\n"
            f"NEVER dump raw JSON to the user. Parse API responses and present them beautifully."
            f"{_append}"
        )

        llm_request = LLMRequest(
            prompt=message,
            system_prompt=system_prompt,
            model_preference=ModelTier.FAST,
            sensitivity=SensitivityLevel.INTERNAL,
            max_tokens=2048,
            temperature=0.7,
        )

        # Build provider list — put the user-selected model first if specified
        providers = self._get_llm_providers()
        if model_override:
            providers = self._resolve_model_override(model_override, providers)

        # Try each configured provider in order
        response_text = None
        provider_errors = []
        for provider_name, api_key, model in providers:
            try:
                client = self._create_llm_client(provider_name, api_key, model_override=model)
                llm_response = await client.complete(llm_request)
                response_text = llm_response.content
                await self._logger.ainfo(
                    "chat_provider_used",
                    agent_id=self.identity.id,
                    provider=provider_name,
                    model=model or "default",
                )
                break
            except Exception as exc:
                error_msg = str(exc)[:200]
                provider_errors.append(f"{provider_name}: {error_msg}")
                await self._logger.awarning(
                    "chat_provider_failed",
                    agent_id=self.identity.id,
                    provider=provider_name,
                    model=model or "default",
                    error=error_msg,
                )
                continue

        # All providers exhausted — use template fallback
        if response_text is None:
            await self._logger.awarning(
                "chat_all_providers_failed",
                agent_id=self.identity.id,
                errors=provider_errors,
            )
            # Record the failure in the error ring buffer and bump the counter
            combined_error = Exception(
                f"All LLM providers failed: {'; '.join(provider_errors)}"
            )
            self._log_error(
                task_id=task.get("id", "chat"),
                error=combined_error,
                context={"task_type": "chat", "providers_tried": len(provider_errors)},
            )
            self._task_count_failed += 1

            error_details = "\n".join(f"  - {e}" for e in provider_errors) if provider_errors else "  No providers attempted."
            response_text = (
                f"I'm **{self.identity.name}** ({self.identity.role} agent). "
                f"I received your message but couldn't reach any LLM backend.\n\n"
                f"**Provider errors:**\n{error_details}\n\n"
                f"Please check your API keys in Settings or start Ollama for local inference."
            )

        return {
            "status": "completed",
            "response": response_text,
            "agent_id": self.identity.id,
        }

    @abstractmethod
    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a task assigned to this agent.

        Implement this method to define agent-specific task processing logic.

        Args:
            task: Task payload containing task description and parameters.

        Returns:
            Task result payload.

        Raises:
            Exception: If task processing fails.
        """
        pass

    def _get_graph_store(self) -> Any:
        """Get the shared graph store from app state (Neo4j or InMemoryGraphStore).

        Returns:
            Graph store instance, or None if not initialized.
        """
        try:
            from src.main import _app_state
            return _app_state.get("graph_store")
        except Exception:
            return None

    async def create_subtask(
        self,
        description: str,
        priority: int = 3,
        payload: Optional[dict[str, Any]] = None,
        required_capabilities: Optional[list[str]] = None,
        parent_task_id: Optional[str] = None,
        wait: bool = False,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Create a subtask and route it to the best available agent.

        Enables agents to decompose complex work into smaller tasks that
        get assigned to specialist agents via the orchestrator.

        Args:
            description: What the subtask should accomplish.
            priority: Priority level (1=critical, 5=deferred).
            payload: Task data/parameters.
            required_capabilities: List of required agent capabilities.
                If empty, the orchestrator picks the best fit.
            parent_task_id: ID of the parent task (for lineage tracking).
            wait: If True, wait for the subtask to complete and return its result.
            timeout: Max seconds to wait when wait=True.

        Returns:
            Dict with task_id and status. If wait=True, also includes the result.

        Raises:
            RuntimeError: If task manager or orchestrator is not wired.
        """
        if not self._task_manager or not self._orchestrator:
            await self._logger.awarning(
                "subtask_creation_unavailable",
                agent_id=self.identity.id,
                reason="task_manager or orchestrator not wired",
            )
            raise RuntimeError(
                "Cannot create subtasks: task manager or orchestrator not connected. "
                "Ensure agents are wired after registration."
            )

        import asyncio

        # Create the task via the task manager
        task_id = await self._task_manager.create_task(
            description=description,
            priority=priority,
            payload={
                **(payload or {}),
                "parent_task_id": parent_task_id,
                "created_by_agent": self.identity.id,
            },
            timeout_seconds=int(timeout),
            max_retries=1,
            tags=["subtask", f"parent:{parent_task_id or 'none'}", f"creator:{self.identity.id}"],
        )

        await self._logger.ainfo(
            "subtask_created",
            agent_id=self.identity.id,
            subtask_id=task_id,
            description=description[:100],
            parent_task_id=parent_task_id,
            required_capabilities=required_capabilities,
        )

        # Route via the orchestrator
        task_data = self._task_manager.get_task(task_id)
        if task_data and required_capabilities:
            task_data["required_capabilities"] = required_capabilities

        agent = self._orchestrator.find_suitable_agent(task_data)
        if agent:
            await self._task_manager.assign_task(task_id, agent.identity.id)

            await self._logger.ainfo(
                "subtask_routed",
                subtask_id=task_id,
                assigned_agent=agent.identity.id,
                agent_name=agent.identity.name,
            )

            if wait:
                # Execute and wait for result
                try:
                    result, success = await asyncio.wait_for(
                        agent._process_task_with_audit(task_data),
                        timeout=timeout,
                    )
                    return {
                        "task_id": task_id,
                        "status": "completed" if success else "failed",
                        "assigned_agent": agent.identity.id,
                        "result": result,
                    }
                except asyncio.TimeoutError:
                    return {
                        "task_id": task_id,
                        "status": "timeout",
                        "assigned_agent": agent.identity.id,
                        "error": f"Subtask timed out after {timeout}s",
                    }
        else:
            await self._logger.awarning(
                "subtask_unroutable",
                subtask_id=task_id,
                required_capabilities=required_capabilities,
            )

        return {
            "task_id": task_id,
            "status": "submitted" if agent else "unroutable",
            "assigned_agent": agent.identity.id if agent else None,
        }

    def discover_agents(
        self,
        capability: Optional[str] = None,
        security_level: int = 1,
    ) -> list["AgentIdentity"]:
        """Discover other available agents via the orchestrator.

        Allows an agent to find out which other agents are online and
        what capabilities they have, enabling intelligent delegation.

        Args:
            capability: Filter by specific capability name.
            security_level: Minimum security level required.

        Returns:
            List of AgentIdentity objects for matching agents.
        """
        if not self._orchestrator:
            return []
        return self._orchestrator.discover_agents(
            capability=capability,
            security_level=security_level,
        )

    async def _process_task_with_audit(
        self, task: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Process a task with audit logging and error handling.

        Args:
            task: Task to process.

        Returns:
            Tuple of (result, success).
        """
        task_id = task.get("id", "unknown")
        self._state = AgentState.BUSY

        try:
            result = await self.process_task(task)
            self._task_count_completed += 1

            audit_log = {
                "task_id": task_id,
                "agent_id": self.identity.id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "signature": self._sign_audit_log(task_id, "completed"),
            }

            await self._logger.ainfo(
                "task_completed",
                task_id=task_id,
                audit_log=audit_log,
            )

            # Store task result in knowledge graph
            graph = self._get_graph_store()
            if graph:
                try:
                    summary = (
                        result.get("response")
                        or result.get("summary")
                        or result.get("status", "completed")
                    )
                    await graph.store_task_result(
                        task_id=task_id,
                        agent_id=self.identity.id,
                        result_summary=str(summary)[:500],
                    )
                except Exception:
                    pass  # Graph storage is best-effort

            self._state = AgentState.IDLE
            return result, True

        except Exception as exc:
            self._task_count_failed += 1
            self._state = AgentState.ERROR

            # Capture in error ring buffer for API/dashboard visibility
            self._log_error(task_id, exc, {"task_type": task.get("type")})

            audit_log = {
                "task_id": task_id,
                "agent_id": self.identity.id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error": str(exc),
                "signature": self._sign_audit_log(task_id, "failed"),
            }

            await self._logger.aerror(
                "task_failed",
                task_id=task_id,
                error=str(exc),
                audit_log=audit_log,
            )

            self._state = AgentState.IDLE
            return {"error": str(exc)}, False

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability.

        Args:
            capability_name: Name of capability to check.

        Returns:
            True if agent has the capability, False otherwise.
        """
        return any(cap.name == capability_name for cap in self.identity.capabilities)

    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get capability details.

        Args:
            capability_name: Name of capability to retrieve.

        Returns:
            AgentCapability object if found, None otherwise.
        """
        for cap in self.identity.capabilities:
            if cap.name == capability_name:
                return cap
        return None

    def report_status(self) -> AgentHeartbeat:
        """Generate a status report for this agent.

        Returns:
            AgentHeartbeat with current status.
        """
        return AgentHeartbeat(
            agent_id=self.identity.id,
            state=self._state,
            uptime_seconds=self.uptime_seconds,
            tasks_completed=self._task_count_completed,
            tasks_failed=self._task_count_failed,
        )

    @staticmethod
    def _sign_audit_log(
        data: str,
        status: str,
        secret: str = "default-secret",
    ) -> str:
        """Sign audit log entry with HMAC-SHA256.

        Args:
            data: Data to sign.
            status: Status being logged.
            secret: Secret key for signing.

        Returns:
            HMAC-SHA256 signature as hex string.
        """
        message = f"{data}:{status}".encode()
        signature = hmac.new(
            secret.encode(),
            message,
            hashlib.sha256,
        ).hexdigest()
        return signature
