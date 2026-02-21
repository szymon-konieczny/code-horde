"""Orchestrator for managing agents and tasks in the Code Horde system."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Optional

import structlog
from pydantic import BaseModel, Field

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent

logger = structlog.get_logger(__name__)


class TaskPriority(int, Enum):
    """Task priority levels.

    Attributes:
        CRITICAL: Highest priority - immediate execution.
        HIGH: High priority - execute soon.
        NORMAL: Standard priority.
        LOW: Low priority - background execution.
        DEFERRED: Deferred - execute when resources available.
    """

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    DEFERRED = 5


class TaskStatus(str, Enum):
    """Task lifecycle status.

    Attributes:
        PENDING: Task created, not yet queued.
        QUEUED: Task in queue, awaiting assignment.
        ASSIGNED: Task assigned to agent.
        IN_PROGRESS: Agent actively processing.
        AWAITING_APPROVAL: Requires human approval before proceeding.
        COMPLETED: Task completed successfully.
        FAILED: Task failed.
        CANCELLED: Task cancelled by user/system.
    """

    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Represents a task to be executed by an agent.

    Attributes:
        id: Unique task identifier.
        description: Human-readable task description.
        priority: Task priority level.
        assigned_agent: ID of assigned agent (optional).
        status: Current task status.
        created_at: Task creation timestamp.
        updated_at: Last update timestamp.
        completed_at: Completion timestamp (if completed).
        payload: Task data/parameters.
        result: Task result (if completed).
        error: Error message (if failed).
        retry_count: Number of retry attempts.
        max_retries: Maximum retry attempts.
        timeout_seconds: Task timeout in seconds.
        parent_task_id: ID of parent task (for nested tasks).
        required_capabilities: Capabilities required to execute.
        security_level: Minimum security level required.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task ID")
    description: str = Field(description="Task description")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    assigned_agent: Optional[str] = Field(default=None, description="Assigned agent ID")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
    payload: dict[str, Any] = Field(default_factory=dict, description="Task data")
    result: Optional[dict[str, Any]] = Field(default=None, description="Task result")
    error: Optional[str] = Field(default=None, description="Error message")
    retry_count: int = Field(default=0, ge=0, description="Current retry count")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retries")
    timeout_seconds: int = Field(default=3600, gt=0, description="Timeout in seconds")
    parent_task_id: Optional[str] = Field(default=None, description="Parent task ID")
    required_capabilities: list[str] = Field(default_factory=list, description="Required capabilities")
    security_level: int = Field(default=1, ge=1, le=5, description="Minimum security level")


class Orchestrator:
    """Orchestrator for managing agents and coordinating task execution.

    The Orchestrator handles:
    - Agent registration and discovery
    - Task routing to appropriate agents
    - Task lifecycle management
    - Workflow execution (sequential, parallel, conditional)
    - Escalation to human operators
    - Health monitoring of all agents

    Attributes:
        agents: Registry of available agents.
        tasks: Registry of all tasks.
        workflows: Registry of defined workflows.
    """

    def __init__(self, whatsapp_webhook_url: Optional[str] = None) -> None:
        """Initialize the Orchestrator.

        Args:
            whatsapp_webhook_url: Optional WhatsApp webhook URL for escalations.
        """
        self.agents: dict[str, BaseAgent] = {}
        self.tasks: dict[str, Task] = {}
        self.workflows: dict[str, list[Task]] = {}
        self.whatsapp_webhook_url = whatsapp_webhook_url
        self._task_queue: asyncio.PriorityQueue[tuple[int, str]] = asyncio.PriorityQueue()
        self._logger = structlog.get_logger(__name__)
        # Skill registry — set during startup for version-aware routing
        self._skill_registry: Optional[Any] = None  # type: Optional[SkillRegistry]

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: Agent to register.

        Raises:
            ValueError: If agent with same ID already registered.
        """
        if agent.identity.id in self.agents:
            raise ValueError(f"Agent {agent.identity.id} already registered")

        await agent.startup()
        self.agents[agent.identity.id] = agent

        await self._logger.ainfo(
            "agent_registered",
            agent_id=agent.identity.id,
            agent_name=agent.identity.name,
            capabilities=[cap.name for cap in agent.identity.capabilities],
        )

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator.

        Args:
            agent_id: ID of agent to unregister.

        Raises:
            KeyError: If agent not found.
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]
        await agent.shutdown()
        del self.agents[agent_id]

        await self._logger.ainfo(
            "agent_unregistered",
            agent_id=agent_id,
            agent_name=agent.identity.name,
        )

    def discover_agents(
        self,
        capability: Optional[str] = None,
        security_level: int = 1,
    ) -> list[AgentIdentity]:
        """Discover available agents, optionally filtered by capability.

        Args:
            capability: Optional capability to filter by.
            security_level: Minimum security level required.

        Returns:
            List of agent identities matching criteria.
        """
        agents = []

        for agent in self.agents.values():
            if agent.state in (AgentState.ERROR, AgentState.OFFLINE):
                continue

            if agent.identity.security_level < security_level:
                continue

            if capability and not agent.has_capability(capability):
                continue

            agents.append(agent.identity)

        return agents

    def discover_agents_with_skills(
        self,
        exclude_agent_id: Optional[str] = None,
    ) -> list[tuple[AgentIdentity, list[AgentCapability]]]:
        """Discover available agents together with their skill lists.

        Used by agents to build delegation-aware system prompts so the
        LLM knows which other agents exist and what they can do.

        Args:
            exclude_agent_id: Agent to exclude (typically the caller).

        Returns:
            List of ``(identity, capabilities)`` tuples for every
            healthy agent.
        """
        result: list[tuple[AgentIdentity, list[AgentCapability]]] = []

        for agent in self.agents.values():
            if agent.state in (AgentState.ERROR, AgentState.OFFLINE):
                continue
            if exclude_agent_id and agent.identity.id == exclude_agent_id:
                continue

            # Prefer registry (has tags/versions), fall back to identity
            skills: list[AgentCapability]
            if self._skill_registry is not None:
                skills = list(self._skill_registry.get_skills(agent.identity.id))
            if not skills:
                skills = list(agent.identity.capabilities)

            result.append((agent.identity, skills))

        return result

    def create_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        payload: Optional[dict[str, Any]] = None,
        required_capabilities: Optional[list[str]] = None,
        timeout_seconds: int = 3600,
        max_retries: int = 3,
    ) -> Task:
        """Create a new task.

        Args:
            description: Task description.
            priority: Task priority level.
            payload: Task data/parameters.
            required_capabilities: List of required capabilities.
            timeout_seconds: Task timeout.
            max_retries: Maximum retry attempts.

        Returns:
            Created Task object.
        """
        task = Task(
            description=description,
            priority=priority,
            payload=payload or {},
            required_capabilities=required_capabilities or [],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        self.tasks[task.id] = task
        return task

    async def route_task(self, task: Task) -> bool:
        """Route a task to the best available agent.

        Routes based on:
        - Required capabilities
        - Agent security level
        - Current agent state
        - Task priority

        Args:
            task: Task to route.

        Returns:
            True if task was successfully assigned, False otherwise.
        """
        # Find candidates with required capabilities
        candidates = []

        for agent in self.agents.values():
            if agent.state not in (AgentState.IDLE, AgentState.PAUSED):
                continue

            if agent.identity.security_level < task.security_level:
                continue

            # Check if agent has all required capabilities
            if not all(agent.has_capability(cap) for cap in task.required_capabilities):
                continue

            candidates.append(agent)

        if not candidates:
            await self._logger.awarning(
                "no_agent_available",
                task_id=task.id,
                required_capabilities=task.required_capabilities,
            )
            return False

        # Sort candidates by how many optional capabilities they have
        # (to use less-capable agents for simpler tasks)
        best_agent = sorted(
            candidates,
            key=lambda a: len(a.identity.capabilities),
        )[0]

        task.assigned_agent = best_agent.identity.id
        task.status = TaskStatus.ASSIGNED
        task.updated_at = datetime.now(timezone.utc)

        await self._task_queue.put((task.priority.value, task.id))

        await self._logger.ainfo(
            "task_assigned",
            task_id=task.id,
            agent_id=best_agent.identity.id,
            agent_name=best_agent.identity.name,
        )

        return True

    async def execute_task(self, task_id: str) -> dict[str, Any]:
        """Execute a task on its assigned agent.

        Args:
            task_id: ID of task to execute.

        Returns:
            Task result.

        Raises:
            KeyError: If task not found.
            ValueError: If task has no assigned agent.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        if not task.assigned_agent:
            raise ValueError(f"Task {task_id} has no assigned agent")

        if task.assigned_agent not in self.agents:
            raise ValueError(f"Agent {task.assigned_agent} not found")

        agent = self.agents[task.assigned_agent]

        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now(timezone.utc)

        try:
            # Execute with timeout
            result, success = await asyncio.wait_for(
                agent._process_task_with_audit(task.payload),
                timeout=task.timeout_seconds,
            )

            if success:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now(timezone.utc)

                await self._logger.ainfo(
                    "task_execution_success",
                    task_id=task.id,
                    agent_id=agent.identity.id,
                    duration_seconds=(task.completed_at - task.created_at).total_seconds(),
                )
            else:
                task.status = TaskStatus.FAILED
                task.error = result.get("error", "Unknown error")
                task.updated_at = datetime.now(timezone.utc)

                await self._logger.aerror(
                    "task_execution_failed",
                    task_id=task.id,
                    agent_id=agent.identity.id,
                    error=task.error,
                )

            return result

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timeout after {task.timeout_seconds} seconds"
            task.updated_at = datetime.now(timezone.utc)

            await self._logger.aerror(
                "task_execution_timeout",
                task_id=task.id,
                agent_id=agent.identity.id,
            )

            if task.retry_count < task.max_retries:
                await self.retry_task(task_id)

            return {"error": task.error}

    async def retry_task(self, task_id: str, delay_seconds: float = 5.0) -> bool:
        """Retry a failed task with exponential backoff.

        Args:
            task_id: ID of task to retry.
            delay_seconds: Initial delay before retry.

        Returns:
            True if retry was scheduled, False otherwise.

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        if task.retry_count >= task.max_retries:
            await self._logger.awarning(
                "task_max_retries_exceeded",
                task_id=task.id,
                max_retries=task.max_retries,
            )
            return False

        # Exponential backoff
        backoff_delay = delay_seconds * (2 ** task.retry_count)

        task.retry_count += 1
        task.status = TaskStatus.QUEUED
        task.updated_at = datetime.now(timezone.utc)

        await self._logger.ainfo(
            "task_retry_scheduled",
            task_id=task.id,
            retry_count=task.retry_count,
            delay_seconds=backoff_delay,
        )

        await asyncio.sleep(backoff_delay)
        return await self.route_task(task)

    async def escalate_to_human(
        self,
        task_id: str,
        reason: str,
        user_phone: Optional[str] = None,
    ) -> bool:
        """Escalate a task to a human operator via WhatsApp.

        Args:
            task_id: ID of task to escalate.
            reason: Reason for escalation.
            user_phone: Phone number to send escalation message.

        Returns:
            True if escalation was sent, False otherwise.

        Raises:
            KeyError: If task not found.
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = TaskStatus.AWAITING_APPROVAL
        task.updated_at = datetime.now(timezone.utc)

        message = (
            f"Task escalation:\n"
            f"ID: {task.id}\n"
            f"Description: {task.description}\n"
            f"Reason: {reason}\n"
            f"Please review and provide guidance."
        )

        await self._logger.ainfo(
            "task_escalated_to_human",
            task_id=task.id,
            reason=reason,
            phone=user_phone,
        )

        # In production, this would call WhatsApp API
        # For now, just log the escalation
        return True

    async def execute_workflow(
        self,
        workflow_name: str,
        workflow_type: str = "sequential",
        tasks: Optional[list[Task]] = None,
    ) -> dict[str, Any]:
        """Execute a workflow of multiple tasks.

        Supports:
        - sequential: Execute tasks one after another
        - parallel: Execute all tasks simultaneously
        - conditional: Execute based on task results

        Args:
            workflow_name: Name of workflow.
            workflow_type: Type of workflow execution.
            tasks: List of tasks in workflow.

        Returns:
            Workflow execution results.

        Raises:
            ValueError: If workflow type is invalid.
        """
        if tasks is None:
            tasks = []

        if workflow_type == "sequential":
            return await self._execute_sequential_workflow(workflow_name, tasks)
        elif workflow_type == "parallel":
            return await self._execute_parallel_workflow(workflow_name, tasks)
        elif workflow_type == "conditional":
            return await self._execute_conditional_workflow(workflow_name, tasks)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    async def _execute_sequential_workflow(
        self,
        workflow_name: str,
        tasks: list[Task],
    ) -> dict[str, Any]:
        """Execute tasks sequentially."""
        results = {}

        for task in tasks:
            if await self.route_task(task):
                result = await self.execute_task(task.id)
                results[task.id] = result
            else:
                results[task.id] = {"error": "Failed to assign task"}
                break

        await self._logger.ainfo(
            "sequential_workflow_completed",
            workflow_name=workflow_name,
            task_count=len(tasks),
        )

        return results

    async def _execute_parallel_workflow(
        self,
        workflow_name: str,
        tasks: list[Task],
    ) -> dict[str, Any]:
        """Execute tasks in parallel."""
        # Route all tasks first
        for task in tasks:
            await self.route_task(task)

        # Execute all tasks concurrently
        execution_tasks = [
            self.execute_task(task.id)
            for task in tasks
            if task.assigned_agent
        ]

        results_list = await asyncio.gather(*execution_tasks, return_exceptions=True)

        results = {}
        for task, result in zip(tasks, results_list):
            if isinstance(result, Exception):
                results[task.id] = {"error": str(result)}
            else:
                results[task.id] = result

        await self._logger.ainfo(
            "parallel_workflow_completed",
            workflow_name=workflow_name,
            task_count=len(tasks),
        )

        return results

    async def _execute_conditional_workflow(
        self,
        workflow_name: str,
        tasks: list[Task],
    ) -> dict[str, Any]:
        """Execute tasks with conditional logic based on results."""
        results = {}

        for task in tasks:
            # Check if this task's condition is met (if parent completed successfully)
            if task.parent_task_id:
                parent_result = results.get(task.parent_task_id)
                if not parent_result or "error" in parent_result:
                    results[task.id] = {"skipped": "Parent task failed"}
                    continue

            if await self.route_task(task):
                result = await self.execute_task(task.id)
                results[task.id] = result
            else:
                results[task.id] = {"error": "Failed to assign task"}

        await self._logger.ainfo(
            "conditional_workflow_completed",
            workflow_name=workflow_name,
            task_count=len(tasks),
        )

        return results

    async def monitor_health(self) -> dict[str, Any]:
        """Monitor health of all registered agents.

        Returns:
            Health status report for all agents.
        """
        health_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_agents": len(self.agents),
            "agents": {},
        }

        for agent_id, agent in self.agents.items():
            heartbeat = agent.report_status()
            health_report["agents"][agent_id] = {
                "name": agent.identity.name,
                "role": agent.identity.role,
                "state": heartbeat.state.value,
                "uptime_seconds": heartbeat.uptime_seconds,
                "tasks_completed": heartbeat.tasks_completed,
                "tasks_failed": heartbeat.tasks_failed,
            }

        return health_report

    def find_suitable_agent(
        self,
        task: Any,
        version_constraints: Optional[dict[str, str]] = None,
    ) -> Optional[BaseAgent]:
        """Find the best available agent for a given task.

        Picks the first IDLE agent that has all required capabilities.
        Falls back to any IDLE agent if no capabilities are specified.

        When *version_constraints* is provided (mapping of capability name to
        semver range like ``{"generate_code": ">=2.0.0"}``), the registry is
        queried for version compatibility.

        Args:
            task: Task dict or Task object.
            version_constraints: Optional per-capability version ranges.

        Returns:
            A suitable BaseAgent, or None if none available.
        """
        # Extract required capabilities from task
        required_caps: list[str] = []
        if isinstance(task, Task):
            required_caps = task.required_capabilities
        elif isinstance(task, dict):
            required_caps = task.get("required_capabilities", [])

        # Merge version constraints from task payload if present
        if version_constraints is None and isinstance(task, dict):
            version_constraints = task.get("version_constraints")

        candidates = []
        for agent in self.agents.values():
            if agent.state in (AgentState.ERROR, AgentState.OFFLINE):
                continue

            # Capability presence check
            if required_caps and not all(
                agent.has_capability(cap) for cap in required_caps
            ):
                continue

            # Version constraint check (requires skill registry)
            if version_constraints and self._skill_registry is not None:
                version_ok = True
                for cap_name, ver_range in version_constraints.items():
                    if not self._skill_registry.agent_has_skill_version(
                        agent.identity.id, cap_name, ver_range
                    ):
                        version_ok = False
                        break
                if not version_ok:
                    continue

            candidates.append(agent)

        if not candidates:
            return None

        # Prefer IDLE agents, then sort by fewest capabilities (simple tasks → simple agents)
        candidates.sort(
            key=lambda a: (0 if a.state == AgentState.IDLE else 1, len(a.identity.capabilities)),
        )
        return candidates[0]

    def validate_task_routing(self, task: Any) -> tuple[bool, list[str]]:
        """Check whether a task can be routed to at least one agent.

        Returns ``(routable, errors)`` where *errors* lists human-readable
        reasons for routing failure.
        """
        required_caps: list[str] = []
        if isinstance(task, Task):
            required_caps = task.required_capabilities
        elif isinstance(task, dict):
            required_caps = task.get("required_capabilities", [])

        if not required_caps:
            return (True, [])

        errors: list[str] = []
        for cap_name in required_caps:
            found = any(a.has_capability(cap_name) for a in self.agents.values())
            if not found:
                errors.append(f"No agent has capability: {cap_name!r}")

        return (len(errors) == 0, errors)

    def get_available_skills(self, agent_id: str) -> list[AgentCapability]:
        """Return all skills for *agent_id* from the registry or identity."""
        if self._skill_registry is not None:
            skills = self._skill_registry.get_skills(agent_id)
            if skills:
                return list(skills)

        agent = self.agents.get(agent_id)
        if agent:
            return list(agent.identity.capabilities)
        return []

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: ID of task to retrieve.

        Returns:
            Task object if found, None otherwise.
        """
        return self.tasks.get(task_id)
