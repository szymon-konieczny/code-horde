"""Code Horde core module - Agent orchestration and task management system."""

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent
from src.core.config import (
    ClaudeSettings,
    DatabaseSettings,
    MasterSettings,
    OllamaSettings,
    RabbitMQSettings,
    RedisSettings,
    SecuritySettings,
    Settings,
    SystemSettings,
    WhatsAppSettings,
)
from src.core.message_bus import Message, MessageBus
from src.core.orchestrator import Orchestrator, Task, TaskPriority
from src.core.reasoning import (
    ChainOfThoughtEngine,
    ReasoningChain,
    ReasoningStrategy,
    ThoughtStep,
)
from src.core.skills import SkillDefinition, SkillDependency, SkillRegistry
from src.core.task_manager import TaskManager, TaskResult, TaskStatus

__all__ = [
    # Configuration
    "SystemSettings",
    "RedisSettings",
    "RabbitMQSettings",
    "DatabaseSettings",
    "ClaudeSettings",
    "OllamaSettings",
    "WhatsAppSettings",
    "SecuritySettings",
    "MasterSettings",
    "Settings",
    # Agent Base
    "AgentState",
    "AgentCapability",
    "AgentIdentity",
    "BaseAgent",
    # Orchestrator
    "TaskPriority",
    "Task",
    "Orchestrator",
    # Message Bus
    "Message",
    "MessageBus",
    # Reasoning
    "ChainOfThoughtEngine",
    "ReasoningChain",
    "ReasoningStrategy",
    "ThoughtStep",
    # Skills
    "SkillDefinition",
    "SkillDependency",
    "SkillRegistry",
    # Task Manager
    "TaskStatus",
    "TaskResult",
    "TaskManager",
]

__version__ = "0.1.0"
