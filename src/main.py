"""FastAPI application and main entry point for AgentArmy system."""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Optional

import pathlib

# Load .env into os.environ so agents can discover all configured LLM keys
from dotenv import load_dotenv
load_dotenv(override=False)

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP

from src.agents import (
    AutomatorAgent,
    BuilderAgent,
    DesignerAgent,
    DevOpsAgent,
    InspectorAgent,
    LinterAgent,
    MarketerAgent,
    SchedulerAgent,
    ScoutAgent,
    ScribeAgent,
    SentinelAgent,
    WatcherAgent,
)
from src.bridges.whatsapp import WhatsAppBridge
from src.bridges.jira import JiraBridge, extract_jira_refs
from src.bridges.figma import FigmaBridge, extract_figma_refs
from src.bridges.github import GitHubBridge, extract_github_refs
from src.bridges.playwright_runner import PlaywrightRunner, detect_playwright_setup
from src.core.agent_base import AgentIdentity, AgentState
from src.core.config import Settings
from src.core.orchestrator import Orchestrator
from src.core.task_manager import TaskManager, TaskStatus
from src.models.router import LLMRouter
from src.platform import detect_platform, get_desktop_adapter, get_calendar_adapter
from src.cluster import WorkerRegistry
from src.storage import Database, RedisStore, InMemoryStore, Neo4jStore, InMemoryGraphStore, HypergraphStore, ConversationStore
from src.storage.memory import AgentMemoryStore, MEMORY_TYPES, MEMORY_SCOPES
from src.api.setup import router as setup_router

logger = structlog.get_logger(__name__)

# Global application state
_app_state: dict[str, Any] = {}
_cluster_mcp: Optional[FastMCP] = None


def _get_mcp_app() -> FastMCP:
    """Get or create the cluster MCP FastMCP app instance."""
    global _cluster_mcp
    if _cluster_mcp is None:
        _cluster_mcp = FastMCP("agentarmy-center")
    return _cluster_mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown.

    Handles:
    - Initialization of all services
    - Agent registration and startup
    - Graceful shutdown and cleanup
    """
    await startup()
    yield
    await shutdown()


async def startup() -> None:
    """Initialize all application services and agents.

    If required configuration (e.g. API keys) is missing, starts in
    *setup mode* — only the dashboard and setup API are available so
    the user can configure keys through the onboarding wizard.

    Raises:
        Exception: If initialization fails for reasons other than
            missing configuration.
    """
    try:
        await logger.ainfo("application_startup_starting")

        # Load configuration — all fields now have safe defaults
        settings = Settings()
        _app_state["settings"] = settings

        # ── Check if we have minimum config to run full system ─────
        if not settings.is_configured:
            await logger.ainfo(
                "application_startup_setup_mode",
                reason="Claude API key not configured — running in setup mode. "
                       "Agents will start but LLM features will use fallback responses.",
            )
            _app_state["setup_mode"] = True
        else:
            _app_state["setup_mode"] = False

        # ── Load project directory from .env ────────────────────────
        import os
        project_dir = os.environ.get("AGENTARMY_PROJECT_DIR", "")
        if project_dir and pathlib.Path(project_dir).is_dir():
            _app_state["project_dir"] = project_dir
            await logger.ainfo("project_directory", path=project_dir)
        else:
            _app_state["project_dir"] = ""

        # ── Initialize services — always start agents ─────────────

        # Initialize storage layer
        await logger.ainfo("initializing_storage_layer")
        try:
            db = Database(
                url=settings.database_url,
                pool_size=settings.database_pool_size,
            )
            await db.connect()
            _app_state["db"] = db
        except Exception as db_exc:
            await logger.awarning("database_connection_skipped", error=str(db_exc))

        try:
            redis = RedisStore(
                url=settings.redis_url,
                default_ttl=settings.redis_ttl,
            )
            await redis.connect()
            _app_state["redis"] = redis
            await logger.ainfo("cache_backend", backend="redis")
        except Exception as redis_exc:
            await logger.ainfo(
                "redis_unavailable_using_memory",
                reason=str(redis_exc),
            )
            # Fall back to in-memory cache — works perfectly for desktop mode
            memory_store = InMemoryStore(default_ttl=settings.redis_ttl)
            await memory_store.connect()
            _app_state["redis"] = memory_store
            await logger.ainfo("cache_backend", backend="in-memory")

        # Initialize knowledge graph — Neo4j with InMemoryGraphStore fallback
        await logger.ainfo("initializing_knowledge_graph")
        try:
            graph_store = Neo4jStore(
                uri=settings.neo4j_bolt_url,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                database=settings.neo4j_database,
            )
            await graph_store.connect()
            _app_state["graph_store"] = graph_store
            await logger.ainfo("graph_backend", backend="neo4j")
        except Exception as neo4j_exc:
            await logger.ainfo(
                "neo4j_unavailable_using_memory",
                reason=str(neo4j_exc),
            )
            memory_graph = InMemoryGraphStore()
            await memory_graph.connect()
            _app_state["graph_store"] = memory_graph
            await logger.ainfo("graph_backend", backend="in-memory")

        # Initialize hypergraph store (collaboration tracking)
        hypergraph = HypergraphStore()
        _app_state["hypergraph_store"] = hypergraph
        await logger.ainfo("hypergraph_store_initialized")

        # Initialize conversation store (persistence for chats & tasks)
        conv_store = ConversationStore()
        _app_state["conversation_store"] = conv_store
        stats = await conv_store.get_stats()
        await logger.ainfo(
            "conversation_store_initialized",
            conversations=stats["conversation_count"],
            messages=stats["total_messages"],
        )

        # Initialize LLM router
        await logger.ainfo("initializing_llm_router")
        llm_router = LLMRouter()
        _app_state["llm_router"] = llm_router

        # Initialize orchestrator and task manager
        await logger.ainfo("initializing_orchestrator")
        task_manager = TaskManager()
        orchestrator = Orchestrator()
        _app_state["task_manager"] = task_manager
        _app_state["orchestrator"] = orchestrator

        # Initialize agents
        await logger.ainfo("initializing_agents")
        agents = []

        sentinel = SentinelAgent(
            agent_id="sentinel-001",
            whatsapp_enabled=settings.whatsapp_enabled,
        )
        await sentinel.startup()
        agents.append(sentinel)

        builder = BuilderAgent(agent_id="builder-001")
        await builder.startup()
        agents.append(builder)

        inspector = InspectorAgent(agent_id="inspector-001")
        await inspector.startup()
        agents.append(inspector)

        watcher = WatcherAgent(agent_id="watcher-001")
        await watcher.startup()
        agents.append(watcher)

        scout = ScoutAgent(agent_id="scout-001")
        await scout.startup()
        agents.append(scout)

        scribe = ScribeAgent(agent_id="scribe-001")
        await scribe.startup()
        agents.append(scribe)

        devops = DevOpsAgent(agent_id="devops-001")
        await devops.startup()
        agents.append(devops)

        marketer = MarketerAgent(agent_id="marketer-001")
        await marketer.startup()
        agents.append(marketer)

        designer = DesignerAgent(agent_id="designer-001")
        await designer.startup()
        agents.append(designer)

        linter = LinterAgent(agent_id="linter-001")
        await linter.startup()
        agents.append(linter)

        automator = AutomatorAgent(agent_id="automator-001", desktop=get_desktop_adapter())
        await automator.startup()
        agents.append(automator)

        scheduler = SchedulerAgent(agent_id="scheduler-001", calendar=get_calendar_adapter())
        await scheduler.startup()
        agents.append(scheduler)

        _app_state["agents"] = {agent.identity.id: agent for agent in agents}

        # Register agents with orchestrator
        await logger.ainfo("registering_agents_with_orchestrator")
        for agent in agents:
            await orchestrator.register_agent(agent)

        # Wire task manager + orchestrator into agents for subtask creation
        for agent in agents:
            agent._task_manager = task_manager
            agent._orchestrator = orchestrator

        # Initialize WhatsApp bridge if enabled and configured
        if settings.whatsapp_enabled:
            try:
                await logger.ainfo("initializing_whatsapp_bridge")
                whatsapp = WhatsAppBridge(
                    access_token=settings.whatsapp_api_token,
                    phone_number_id=settings.whatsapp_phone_number_id,
                    business_account_id=settings.whatsapp_phone_number_id,
                )
                _app_state["whatsapp"] = whatsapp
            except Exception as wa_exc:
                await logger.awarning("whatsapp_bridge_skipped", error=str(wa_exc))

        # Initialize Jira + Figma + GitHub bridges (lazy — they read env vars on use)
        _app_state["jira_bridge"] = JiraBridge()
        _app_state["figma_bridge"] = FigmaBridge()
        _app_state["github_bridge"] = GitHubBridge()
        if _app_state["jira_bridge"].is_configured:
            await logger.ainfo("jira_bridge_ready")
        if _app_state["figma_bridge"].is_configured:
            await logger.ainfo("figma_bridge_ready")
        if _app_state["github_bridge"].is_configured:
            await logger.ainfo("github_bridge_ready")

        # Initialize Playwright runner + auto-provision Chromium
        pw_runner = PlaywrightRunner(project_dir=_app_state.get("project_dir") or None)
        _app_state["playwright_runner"] = pw_runner
        pw_provision = await pw_runner.ensure_chromium()
        await logger.ainfo(
            "playwright_runner_initialized",
            chromium_ready=pw_runner.chromium_ready,
            provision_status=pw_provision.get("status"),
            provision_step=pw_provision.get("step", ""),
        )

        # Initialize persistent memory store
        mem_store = AgentMemoryStore(project_dir=_app_state.get("project_dir") or None)
        _app_state["memory_store"] = mem_store
        mem_stats = mem_store.get_stats()
        await logger.ainfo("memory_store_initialized", total_memories=mem_stats["total"])

        # Start agent heartbeat monitoring
        await logger.ainfo("starting_agent_heartbeats")
        _app_state["heartbeat_task"] = asyncio.create_task(
            _start_heartbeat_monitoring(orchestrator)
        )

        # ── Cluster mode initialization ─────────────────────────────
        cluster_mode = settings.cluster.mode if hasattr(settings, 'cluster') else "standalone"
        _app_state["cluster_mode"] = cluster_mode

        if cluster_mode == "center":
            from src.cluster.center_tools import register_center_tools

            worker_registry = WorkerRegistry()
            _app_state["worker_registry"] = worker_registry
            register_center_tools(
                app=_get_mcp_app(),
                registry=worker_registry,
                token=settings.cluster.center_token.get_secret_value() if settings.cluster.center_token else "",
            )

            # Start periodic stale-worker cleanup
            async def _cleanup_stale_workers():
                while True:
                    await asyncio.sleep(settings.cluster.stale_timeout)
                    try:
                        stale = await worker_registry.cleanup_stale(
                            timeout_seconds=settings.cluster.stale_timeout,
                        )
                        if stale:
                            await logger.ainfo("stale_workers_cleaned", count=len(stale))
                    except Exception as e:
                        await logger.awarning("stale_cleanup_error", error=str(e))

            _app_state["stale_cleanup_task"] = asyncio.create_task(_cleanup_stale_workers())
            await logger.ainfo("cluster_center_initialized", workers=0)

        elif cluster_mode == "worker":
            from src.cluster.worker_client import WorkerClient

            worker_client = WorkerClient(
                center_url=settings.cluster.center_url,
                center_token=settings.cluster.center_token.get_secret_value() if settings.cluster.center_token else "",
                worker_name=settings.cluster.worker_name,
                worker_port=settings.cluster.worker_port,
                heartbeat_interval=settings.cluster.heartbeat_interval,
                agents=list(_app_state.get("agents", {}).values()),
                task_manager=task_manager,
            )
            _app_state["worker_client"] = worker_client
            await worker_client.start()
            await logger.ainfo(
                "cluster_worker_started",
                center_url=settings.cluster.center_url,
                worker_name=worker_client.worker_name,
            )

        await logger.ainfo(
            "application_startup_complete",
            agents_registered=len(agents),
            storage_ready=True,
        )

    except Exception as exc:
        await logger.aerror(
            "application_startup_failed",
            error=str(exc),
        )
        raise


async def shutdown() -> None:
    """Shutdown application and cleanup resources.

    Raises:
        Exception: If shutdown fails.
    """
    try:
        await logger.ainfo("application_shutdown_starting")

        # Cancel heartbeat monitoring
        if "heartbeat_task" in _app_state:
            _app_state["heartbeat_task"].cancel()
            try:
                await _app_state["heartbeat_task"]
            except asyncio.CancelledError:
                pass

        # Cancel cluster tasks
        if "stale_cleanup_task" in _app_state:
            _app_state["stale_cleanup_task"].cancel()
            try:
                await _app_state["stale_cleanup_task"]
            except asyncio.CancelledError:
                pass

        if "worker_client" in _app_state:
            await _app_state["worker_client"].stop()

        # Shutdown all agents
        if "agents" in _app_state:
            await logger.ainfo("shutting_down_agents")
            for agent in _app_state["agents"].values():
                try:
                    await agent.shutdown()
                except Exception as exc:
                    await logger.aerror(
                        "agent_shutdown_failed",
                        agent_id=agent.identity.id,
                        error=str(exc),
                    )

        # Disconnect storage layer
        if "db" in _app_state:
            await logger.ainfo("disconnecting_database")
            try:
                await _app_state["db"].disconnect()
            except Exception as exc:
                await logger.aerror(
                    "database_disconnection_failed",
                    error=str(exc),
                )

        if "redis" in _app_state:
            await logger.ainfo("disconnecting_redis")
            try:
                await _app_state["redis"].disconnect()
            except Exception as exc:
                await logger.aerror(
                    "redis_disconnection_failed",
                    error=str(exc),
                )

        if "graph_store" in _app_state:
            await logger.ainfo("disconnecting_graph_store")
            try:
                await _app_state["graph_store"].disconnect()
            except Exception as exc:
                await logger.aerror(
                    "graph_store_disconnection_failed",
                    error=str(exc),
                )

        await logger.ainfo("application_shutdown_complete")

    except Exception as exc:
        await logger.aerror(
            "application_shutdown_failed",
            error=str(exc),
        )
        raise


async def _start_heartbeat_monitoring(orchestrator: Orchestrator) -> None:
    """Monitor agent heartbeats periodically.

    Args:
        orchestrator: Orchestrator instance to check agent status.
    """
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds

            heartbeats = {}
            for agent_id, agent in _app_state.get("agents", {}).items():
                heartbeat = agent.report_status()
                heartbeats[agent_id] = heartbeat

                await logger.adebug(
                    "agent_heartbeat",
                    agent_id=agent_id,
                    state=heartbeat.state.value,
                    uptime_seconds=heartbeat.uptime_seconds,
                    tasks_completed=heartbeat.tasks_completed,
                )

                # Store heartbeat in Redis
                redis = _app_state.get("redis")
                if redis:
                    await redis.set(
                        f"heartbeat:{agent_id}",
                        heartbeat.model_dump(),
                        ttl=60,
                    )

    except asyncio.CancelledError:
        await logger.ainfo("heartbeat_monitoring_cancelled")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="AgentArmy",
        description="Multi-agent autonomous system for development, security, and operations",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow dashboard and CLI clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Setup / Settings API ─────────────────────────────────────
    app.include_router(setup_router)

    # ── Dashboard UI ─────────────────────────────────────────────
    _dashboard_html: Optional[str] = None

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def serve_dashboard() -> HTMLResponse:
        """Serve the Command Center dashboard."""
        nonlocal _dashboard_html
        if _dashboard_html is None:
            html_path = pathlib.Path(__file__).parent / "dashboard" / "index.html"
            _dashboard_html = html_path.read_text(encoding="utf-8")
        return HTMLResponse(_dashboard_html)

    # ── Static files for generated images ─────────────────────────
    _generated_dir = pathlib.Path(__file__).parent / "generated"
    _generated_dir.mkdir(exist_ok=True)
    app.mount("/generated", StaticFiles(directory=str(_generated_dir)), name="generated")

    # ── Active task tracking for cancellation ───────────────────
    _app_state["active_tasks"] = {}  # task_id → asyncio.Task

    # ── Chat endpoint (used by CLI + Dashboard) ──────────────────
    @app.post("/chat")
    async def chat_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        """Send a free-form message to the orchestrator.

        Awaits the agent's actual response instead of fire-and-forget.

        Args:
            payload: {"message": "...", "agent_id": "..." (optional)}

        Returns:
            {"response": "...", "agent_id": "...", "task_id": "..."}
        """
        message = payload.get("message", "").strip()
        target_agent = payload.get("agent_id")
        model_override = payload.get("model")  # e.g. "claude-sonnet-4-20250514"
        conversation_id = payload.get("conversation_id")

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Persist user message to conversation store
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if conv_store and conversation_id:
            await conv_store.add_message(conversation_id, "user", message)

        orchestrator = _app_state.get("orchestrator")
        task_manager = _app_state.get("task_manager")

        if not orchestrator or not task_manager:
            raise HTTPException(status_code=503, detail="Services not initialized")

        try:
            # ── Resolve Jira/Figma URLs in the message ───────────
            url_context_parts: list[str] = []
            jira_refs = extract_jira_refs(message)
            figma_refs = extract_figma_refs(message)

            if jira_refs:
                jira: JiraBridge = _app_state.get("jira_bridge") or JiraBridge()
                for ref in jira_refs:
                    issue = await jira.fetch_issue(ref["domain"], ref["issue_key"])
                    ctx = jira.format_for_agent_context(issue)
                    if ctx:
                        url_context_parts.append(ctx)

            if figma_refs:
                figma: FigmaBridge = _app_state.get("figma_bridge") or FigmaBridge()
                for ref in figma_refs:
                    try:
                        design = await figma.fetch_file_meta(ref["file_key"], ref.get("node_id"))
                        ctx = figma.format_for_agent_context(design)
                        if ctx:
                            url_context_parts.append(ctx)
                    except Exception as figma_exc:
                        await logger.awarning("figma_resolve_failed", error=str(figma_exc)[:200])

            github_refs = extract_github_refs(message)
            if github_refs:
                gh: GitHubBridge = _app_state.get("github_bridge") or GitHubBridge()
                for ref in github_refs:
                    try:
                        repo_info = await gh.fetch_repo_info(ref["owner"], ref["repo"])
                        ctx = gh.format_for_agent_context(repo_info)
                        if ctx:
                            url_context_parts.append(ctx)
                    except Exception as gh_exc:
                        await logger.awarning("github_resolve_failed", error=str(gh_exc)[:200])

            # Create a task from the chat message
            task_payload: dict[str, Any] = {"source": "chat", "agent_id": target_agent}
            if model_override:
                task_payload["model"] = model_override
            project_dir = _app_state.get("project_dir", "")
            if project_dir:
                task_payload["project_dir"] = project_dir

            # Inject resolved URL context so the agent can reference it
            if url_context_parts:
                task_payload["url_context"] = "\n\n---\n\n".join(url_context_parts)

            # Inject relevant memories so the agent has cross-session knowledge
            mem_store: Optional[AgentMemoryStore] = _app_state.get("memory_store")
            if mem_store:
                memory_ctx = mem_store.format_for_agent_context(query=message, limit=15)
                if memory_ctx:
                    task_payload["memory_context"] = memory_ctx

            # Inject configured integration info so agents know what's available
            integrations_ctx = _build_integrations_context()
            if integrations_ctx:
                task_payload["integrations_context"] = integrations_ctx

            task_id = await task_manager.create_task(
                description=message,
                priority=3,
                payload=task_payload,
                timeout_seconds=120,
                max_retries=1,
                tags=["chat"],
            )

            task = task_manager.get_task(task_id)

            # ── Route to agent(s) ─────────────────────────────────
            agents_list: list[Any] = []

            if target_agent:
                # Explicit agent override — single-agent mode
                explicit = _app_state.get("agents", {}).get(target_agent)
                if explicit:
                    agents_list = [explicit]

            if not agents_list:
                # Smart routing — may return 1+ agents for parallel dispatch
                agents_list = _route_chat_to_agents(
                    message, _app_state.get("agents", {})
                )

            # Last-resort fallback via orchestrator
            if not agents_list:
                fallback = orchestrator.find_suitable_agent(task)
                if fallback:
                    agents_list = [fallback]

            if not agents_list:
                response_text = "No agent is available to handle your request right now."
                return {
                    "response": response_text,
                    "agent_id": "system",
                    "task_id": task_id,
                    "cancelled": False,
                }

            # ── Multi-agent parallel path ─────────────────────────
            if len(agents_list) >= 2:
                return await _handle_multi_agent_chat(
                    message=message,
                    agents_list=agents_list,
                    task_manager=task_manager,
                    task_payload=task_payload,
                    conv_store=conv_store,
                    conversation_id=conversation_id,
                    active_tasks=_app_state["active_tasks"],
                )

            # ── Single-agent path (existing behaviour) ────────────
            agent = agents_list[0]
            agent_name = agent.identity.id

            await task_manager.assign_task(task_id, agent.identity.id)
            await task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)

            # Create an explicit asyncio.Task so it can be cancelled externally
            processing = asyncio.create_task(
                agent._process_task_with_audit(task)
            )
            _app_state["active_tasks"][task_id] = processing

            try:
                result, success = await asyncio.wait_for(
                    asyncio.shield(processing),
                    timeout=180.0,
                )

                if success:
                    await task_manager.complete_task(task_id, result)
                    response_text = (
                        result.get("response")
                        or result.get("summary")
                        or result.get("status", "Task completed.")
                    )
                else:
                    error_msg = result.get("error", "Processing failed")
                    await task_manager.fail_task(task_id, error_msg)
                    response_text = f"I encountered an issue: {error_msg}"

            except asyncio.TimeoutError:
                processing.cancel()
                await task_manager.fail_task(task_id, "Response timeout")
                response_text = "The request timed out. Please try again with a simpler query."

            except asyncio.CancelledError:
                await task_manager.cancel_task(task_id)
                response_text = "Task was stopped."

            finally:
                _app_state["active_tasks"].pop(task_id, None)

            # Persist assistant response to conversation store
            if conv_store and conversation_id and response_text != "Task was stopped.":
                await conv_store.add_message(
                    conversation_id, "assistant", response_text, agent=agent_name
                )

            return {
                "response": response_text,
                "agent_id": agent_name,
                "task_id": task_id,
                "cancelled": response_text == "Task was stopped.",
            }

        except Exception as exc:
            await logger.aerror("chat_processing_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Chat cancel endpoint ─────────────────────────────────────
    @app.post("/chat/cancel")
    async def cancel_chat() -> dict[str, Any]:
        """Cancel all active chat tasks.

        Returns:
            Dictionary with list of cancelled task IDs.
        """
        active = _app_state.get("active_tasks", {})
        cancelled = []
        for task_id, atask in list(active.items()):
            atask.cancel()
            cancelled.append(task_id)
            # Update task manager status
            task_manager = _app_state.get("task_manager")
            if task_manager:
                try:
                    await task_manager.cancel_task(task_id)
                except (KeyError, ValueError):
                    pass
        return {"cancelled": cancelled, "count": len(cancelled)}

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint.

        Returns:
            Dictionary with health status of all components.
        """
        setup_mode = _app_state.get("setup_mode", True)

        db = _app_state.get("db")
        redis = _app_state.get("redis")
        graph = _app_state.get("graph_store")

        db_healthy = False
        if db:
            try:
                db_healthy = await db.health_check()
            except Exception:
                pass

        redis_healthy = False
        if redis:
            try:
                redis_healthy = await redis.health_check()
            except Exception:
                pass

        graph_healthy = False
        graph_backend = "none"
        if graph:
            try:
                graph_healthy = await graph.health_check()
                graph_backend = "neo4j" if isinstance(graph, Neo4jStore) else "in-memory"
            except Exception:
                pass

        return {
            "status": "setup" if setup_mode else "healthy",
            "version": "1.0.0",
            "setup_mode": setup_mode,
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "redis": f"healthy ({('redis' if isinstance(redis, RedisStore) else 'in-memory')})" if redis_healthy else "unhealthy",
                "graph": f"healthy ({graph_backend})" if graph_healthy else "unhealthy",
                "agents": len(_app_state.get("agents", {})),
            },
        }

    # WhatsApp webhook verification (GET)
    @app.get("/webhook/whatsapp")
    async def verify_whatsapp_webhook(
        hub_mode: Optional[str] = None,
        hub_challenge: Optional[str] = None,
        hub_verify_token: Optional[str] = None,
    ) -> str:
        """WhatsApp webhook verification endpoint.

        Args:
            hub_mode: Webhook mode (should be 'subscribe')
            hub_challenge: Challenge token to verify
            hub_verify_token: Verification token

        Returns:
            Challenge token if verification successful

        Raises:
            HTTPException: If verification fails
        """
        settings = _app_state.get("settings")

        if (
            hub_mode == "subscribe"
            and hub_verify_token == settings.whatsapp_verify_token
        ):
            await logger.ainfo("whatsapp_webhook_verified")
            return hub_challenge

        await logger.awarning(
            "whatsapp_webhook_verification_failed",
            mode=hub_mode,
        )
        raise HTTPException(status_code=403, detail="Verification failed")

    # WhatsApp message webhook (POST)
    @app.post("/webhook/whatsapp")
    async def handle_whatsapp_webhook(request: Request) -> JSONResponse:
        """Handle incoming WhatsApp messages.

        Args:
            request: HTTP request with WhatsApp payload

        Returns:
            JSON response confirming receipt

        Raises:
            HTTPException: If processing fails
        """
        try:
            payload = await request.json()

            await logger.ainfo(
                "whatsapp_webhook_received",
                payload_type=payload.get("object"),
            )

            # Process WhatsApp message
            whatsapp = _app_state.get("whatsapp")
            if whatsapp:
                await whatsapp.process_message(payload)

            return JSONResponse({"status": "ok"})

        except Exception as exc:
            await logger.aerror(
                "whatsapp_webhook_processing_failed",
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail="Processing failed")

    # WhatsApp test send endpoint
    @app.post("/webhook/whatsapp/test")
    async def test_whatsapp_send(request: Request) -> dict[str, Any]:
        """Send a test WhatsApp message.

        Expects JSON body: {"to": "phone_number", "message": "text"}

        Args:
            request: HTTP request with test payload.

        Returns:
            Result of the send attempt.
        """
        whatsapp = _app_state.get("whatsapp")
        if not whatsapp:
            raise HTTPException(
                status_code=503,
                detail="WhatsApp bridge not initialized. Check ENABLE_WHATSAPP and API token in settings.",
            )

        payload = await request.json()
        to_number = payload.get("to", "").strip().replace("+", "").replace("-", "").replace(" ", "")
        message = payload.get("message", "Hello from AgentArmy!")

        if not to_number:
            raise HTTPException(status_code=400, detail="Missing 'to' phone number")

        try:
            message_id = await whatsapp.send_text(
                to_number=to_number,
                text=message,
            )
            if message_id:
                await logger.ainfo(
                    "whatsapp_test_message_sent",
                    to_number=to_number,
                    message_id=message_id,
                )
                return {
                    "status": "sent",
                    "message_id": message_id,
                    "to": to_number,
                }
            else:
                return {"status": "failed", "detail": "Rate limited or send failed"}
        except Exception as exc:
            await logger.aerror("whatsapp_test_send_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Send failed: {exc}")

    # ── URL Resolution endpoint ───────────────────────────────────
    @app.post("/resolve-urls")
    async def resolve_urls(request: Request) -> dict[str, Any]:
        """Resolve Jira and Figma URLs in a message to rich metadata.

        Body: {"text": "Check https://myco.atlassian.net/browse/PROJ-123 and this figma..."}

        Returns list of resolved items with source, metadata, and formatted display.
        """
        payload = await request.json()
        text = payload.get("text", "")

        results: list[dict[str, Any]] = []

        # Resolve Jira URLs
        jira_refs = extract_jira_refs(text)
        if jira_refs:
            jira: JiraBridge = _app_state.get("jira_bridge") or JiraBridge()
            for ref in jira_refs:
                issue = await jira.fetch_issue(ref["domain"], ref["issue_key"])
                results.append({
                    "source": "jira",
                    "ref": ref,
                    "data": issue,
                    "card_html": jira.format_for_chat(issue),
                    "agent_context": jira.format_for_agent_context(issue),
                })

        # Resolve Figma URLs
        figma_refs = extract_figma_refs(text)
        if figma_refs:
            figma: FigmaBridge = _app_state.get("figma_bridge") or FigmaBridge()
            for ref in figma_refs:
                try:
                    design = await figma.fetch_file_meta(ref["file_key"], ref.get("node_id"))
                    results.append({
                        "source": "figma",
                        "ref": ref,
                        "data": design,
                        "card_html": figma.format_for_chat(design),
                        "agent_context": figma.format_for_agent_context(design),
                    })
                except Exception as exc:
                    results.append({
                        "source": "figma",
                        "ref": ref,
                        "data": {"error": f"Failed to fetch: {str(exc)[:100]}"},
                        "card_html": f"**Figma:** Could not load — {str(exc)[:80]}",
                        "agent_context": "",
                    })

        # Resolve GitHub URLs
        github_refs = extract_github_refs(text)
        if github_refs:
            gh: GitHubBridge = _app_state.get("github_bridge") or GitHubBridge()
            for ref in github_refs:
                try:
                    repo_info = await gh.fetch_repo_info(ref["owner"], ref["repo"])
                    results.append({
                        "source": "github",
                        "ref": ref,
                        "data": repo_info,
                        "card_html": gh.format_for_chat(repo_info),
                        "agent_context": gh.format_for_agent_context(repo_info),
                    })
                except Exception as exc:
                    results.append({
                        "source": "github",
                        "ref": ref,
                        "data": {"error": f"Failed to fetch: {str(exc)[:100]}"},
                        "card_html": f"**GitHub:** Could not load — {str(exc)[:80]}",
                        "agent_context": "",
                    })

        return {"resolved": results, "count": len(results)}

    # ── Clone GitHub template endpoint ──────────────────────────────
    @app.post("/github/clone")
    async def clone_github_template(request: Request) -> dict[str, Any]:
        """Clone a GitHub repo as a new project from a template/boilerplate.

        Body: {"owner": "vercel", "repo": "next.js", "name": "my-app", "branch": "canary"}

        Clones into the configured project directory's parent.
        """
        payload = await request.json()
        owner = payload.get("owner", "").strip()
        repo = payload.get("repo", "").strip()
        new_name = payload.get("name", "").strip() or None
        branch = payload.get("branch", "").strip() or None

        if not owner or not repo:
            raise HTTPException(status_code=400, detail="owner and repo are required")

        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            raise HTTPException(
                status_code=400,
                detail="No project directory configured. Set it first so we know where to clone.",
            )

        # Clone into the project directory's parent (sibling folder)
        import pathlib
        target_parent = str(pathlib.Path(project_dir).parent)

        gh: GitHubBridge = _app_state.get("github_bridge") or GitHubBridge()
        result = await gh.clone_template(owner, repo, target_parent, new_name, branch)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    # ── Post-change verification endpoint ────────────────────────
    @app.post("/verify")
    async def verify_project(request: Request) -> dict[str, Any]:
        """Run build/lint verification on the project after code changes.

        Auto-detects the project type and runs the appropriate build command.
        Body: {"command": "auto"} or {"command": "npm run build"} for explicit.

        Returns:
            Status dict with stdout, stderr, success flag.
        """
        import subprocess

        payload = await request.json()
        command = payload.get("command", "auto").strip()

        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            raise HTTPException(
                status_code=400,
                detail="No project directory configured.",
            )

        root = pathlib.Path(project_dir)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail="Project directory does not exist.")

        # Auto-detect build/lint commands based on project files
        if command == "auto":
            commands = []

            if (root / "package.json").is_file():
                # Node.js project — try build then lint
                if (root / "node_modules").is_dir():
                    commands.append(["npm", "run", "build", "--if-present"])
                else:
                    commands.append(["npm", "install"])
                    commands.append(["npm", "run", "build", "--if-present"])

            elif (root / "Cargo.toml").is_file():
                commands.append(["cargo", "check"])

            elif (root / "pyproject.toml").is_file() or (root / "setup.py").is_file():
                commands.append(["python", "-m", "py_compile", "*.py"])

            elif (root / "Makefile").is_file():
                commands.append(["make", "check"])

            elif (root / "go.mod").is_file():
                commands.append(["go", "build", "./..."])

            else:
                return {"status": "skipped", "message": "No recognizable build system found.", "checks": []}
        else:
            # Explicit command — split it into args
            import shlex
            commands = [shlex.split(command)]

        results = []
        overall_success = True

        for cmd in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(root),
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
                success = proc.returncode == 0
                if not success:
                    overall_success = False
                results.append({
                    "command": " ".join(cmd),
                    "success": success,
                    "exit_code": proc.returncode,
                    "stdout": stdout.decode("utf-8", errors="replace")[-2000:] if stdout else "",
                    "stderr": stderr.decode("utf-8", errors="replace")[-2000:] if stderr else "",
                })
                # Stop on first failure
                if not success:
                    break
            except asyncio.TimeoutError:
                overall_success = False
                results.append({
                    "command": " ".join(cmd),
                    "success": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "Command timed out after 120 seconds.",
                })
                break
            except FileNotFoundError:
                results.append({
                    "command": " ".join(cmd),
                    "success": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Command not found: {cmd[0]}",
                })
                break

        # ── Optionally also run Playwright E2E tests ────────────────
        run_e2e = payload.get("e2e", False)
        e2e_result = None
        if run_e2e and overall_success:
            setup = detect_playwright_setup(str(root))
            if setup["installed"] and setup["test_files"]:
                runner: Optional[PlaywrightRunner] = _app_state.get("playwright_runner")
                if runner:
                    runner.set_project_dir(str(root))
                    e2e_result = await runner.run_tests()
                    if e2e_result.get("status") != "pass":
                        overall_success = False

        return {
            "status": "pass" if overall_success else "fail",
            "checks": results,
            **({"e2e": e2e_result} if e2e_result else {}),
        }

    # Apply approved code to a project file
    @app.post("/approve")
    async def approve_code(request: Request) -> dict[str, Any]:
        """Write approved code to a file in the project directory.

        Body: {"filePath": "src/foo.ts", "code": "file contents..."}

        Creates parent directories if needed.  Only writes within the
        configured project directory for safety.

        Returns:
            Status dict with written file path.
        """
        import pathlib

        payload = await request.json()
        file_path = payload.get("filePath", "").strip()
        code = payload.get("code", "")

        if not file_path:
            raise HTTPException(status_code=400, detail="Missing filePath")

        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            raise HTTPException(
                status_code=400,
                detail="No project directory configured. Set it in the Project bar first.",
            )

        root = pathlib.Path(project_dir).resolve()
        target = (root / file_path).resolve()

        # Safety: ensure target is within project dir
        if not str(target).startswith(str(root)):
            raise HTTPException(status_code=403, detail="Path escapes project directory")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            await logger.ainfo("code_approved_and_written", file=str(target))
            return {
                "status": "written",
                "path": str(target),
                "relative": file_path,
            }
        except Exception as exc:
            await logger.aerror("approve_write_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Write failed: {exc}")

    # Agent status endpoint
    @app.get("/agents")
    async def get_agents() -> dict[str, Any]:
        """Get status of all registered agents.

        Returns:
            Dictionary with agent statuses.
        """
        agents = _app_state.get("agents", {})
        agent_statuses = {}

        for agent_id, agent in agents.items():
            heartbeat = agent.report_status()
            agent_statuses[agent_id] = {
                "name": agent.identity.name,
                "role": agent.identity.role,
                "state": heartbeat.state.value,
                "uptime_seconds": heartbeat.uptime_seconds,
                "tasks_completed": heartbeat.tasks_completed,
                "tasks_failed": heartbeat.tasks_failed,
                "capabilities": [
                    {
                        "name": cap.name,
                        "version": cap.version,
                        "description": cap.description,
                    }
                    for cap in agent.identity.capabilities
                ],
            }

        return {
            "agents": agent_statuses,
            "total_agents": len(agents),
        }

    # ── Cluster workers endpoint (center mode only) ───────────
    @app.get("/api/cluster/workers")
    async def get_cluster_workers(include_offline: bool = False) -> dict[str, Any]:
        """Get connected cluster workers (center mode only).

        Args:
            include_offline: Include offline workers in the response.

        Returns:
            Dictionary with workers list and cluster health.
        """
        registry = _app_state.get("worker_registry")
        if not registry:
            return {
                "mode": _app_state.get("cluster_mode", "standalone"),
                "workers": [],
                "health": {},
            }

        workers = registry.list_workers(include_offline=include_offline)
        return {
            "mode": "center",
            "workers": [w.to_dict() for w in workers],
            "health": registry.cluster_health(),
        }

    # Task list endpoint
    @app.get("/tasks")
    async def list_tasks(
        status: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """List all tasks with optional status filter.

        Args:
            status: Filter by task status (pending, in_progress, completed, failed).
            limit: Maximum number of tasks to return.

        Returns:
            Dictionary with tasks list and count.
        """
        task_manager = _app_state.get("task_manager")
        if not task_manager:
            raise HTTPException(status_code=503, detail="Service not available")

        tasks_list = []
        for tid, t in list(task_manager.tasks.items())[:limit]:
            t_status = t.get("status")
            status_val = t_status.value if hasattr(t_status, "value") else str(t_status)

            # Filter by status if requested
            if status and status_val != status:
                continue

            # Extract a human-readable response from the result payload
            result_data = t.get("result")
            result_text = None
            if result_data and isinstance(result_data, dict):
                result_text = (
                    result_data.get("response")
                    or result_data.get("summary")
                    or result_data.get("status")
                )

            tasks_list.append({
                "task_id": tid,
                "description": t.get("description", ""),
                "status": status_val,
                "priority": t.get("priority", 3),
                "assigned_agent": t.get("assigned_agent"),
                "created_at": t.get("created_at").isoformat() if t.get("created_at") else None,
                "tags": t.get("tags", []),
                "error": t.get("error"),
                "result": result_text,
            })

        # Sort: newest first
        tasks_list.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return {
            "tasks": tasks_list,
            "total": len(tasks_list),
        }

    # Task submission endpoint
    @app.post("/tasks")
    async def submit_task(task_data: dict[str, Any]) -> dict[str, Any]:
        """Submit a new task for processing.

        Args:
            task_data: Task payload with description, priority, context

        Returns:
            Dictionary with task ID and status

        Raises:
            HTTPException: If task submission fails
        """
        try:
            task_manager = _app_state.get("task_manager")
            orchestrator = _app_state.get("orchestrator")

            if not task_manager or not orchestrator:
                raise RuntimeError("Services not initialized")

            # Create task
            task_id = await task_manager.create_task(
                description=task_data.get("description", ""),
                priority=task_data.get("priority", 3),
                payload=task_data.get("context", {}),
                timeout_seconds=task_data.get("timeout_seconds", 3600),
                max_retries=task_data.get("max_retries", 3),
                tags=task_data.get("tags", []),
            )

            await logger.ainfo(
                "task_submitted",
                task_id=task_id,
                priority=task_data.get("priority", 3),
            )

            # Assign and process task
            task = task_manager.get_task(task_id)
            agent = orchestrator.find_suitable_agent(task)

            if agent:
                await task_manager.assign_task(task_id, agent.identity.id)
                bg_task = asyncio.create_task(_process_task(task_manager, agent, task))
                _app_state["active_tasks"][task_id] = bg_task
                # Auto-cleanup when done
                bg_task.add_done_callback(lambda _t, _tid=task_id: _app_state.get("active_tasks", {}).pop(_tid, None))

            return {
                "task_id": task_id,
                "status": "submitted",
                "assigned_agent": agent.identity.id if agent else None,
            }

        except Exception as exc:
            await logger.aerror(
                "task_submission_failed",
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc))

    # Task status endpoint
    @app.get("/tasks/{task_id}")
    async def get_task_status(task_id: str) -> dict[str, Any]:
        """Get status of a specific task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with task status and details

        Raises:
            HTTPException: If task not found
        """
        task_manager = _app_state.get("task_manager")

        if not task_manager:
            raise HTTPException(status_code=503, detail="Service not available")

        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "task_id": task_id,
            "description": task.get("description"),
            "status": task.get("status").value if task.get("status") else None,
            "priority": task.get("priority"),
            "assigned_agent": task.get("assigned_agent"),
            "created_at": task.get("created_at").isoformat() if task.get("created_at") else None,
            "result": task.get("result"),
            "error": task.get("error"),
        }

    # Delete task endpoint
    @app.delete("/tasks/{task_id}")
    async def delete_task(task_id: str) -> dict[str, Any]:
        """Delete a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Dictionary confirming deletion

        Raises:
            HTTPException: If task not found
        """
        task_manager = _app_state.get("task_manager")

        if not task_manager:
            raise HTTPException(status_code=503, detail="Service not available")

        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Remove from task manager
        del task_manager.tasks[task_id]

        # Also remove from history if present
        task_manager.task_history.pop(task_id, None)

        return {"status": "deleted", "task_id": task_id}

    # ── Conversation CRUD endpoints ─────────────────────────────
    @app.get("/conversations")
    async def list_conversations() -> dict[str, Any]:
        """List all conversations."""
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if not conv_store:
            return {"conversations": [], "active_id": None}
        convs = await conv_store.list_conversations()
        active_id = await conv_store.get_active_conversation_id()
        return {"conversations": convs, "active_id": active_id}

    @app.post("/conversations")
    async def create_conversation(payload: dict[str, Any] = {}) -> dict[str, Any]:
        """Create a new conversation."""
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if not conv_store:
            raise HTTPException(status_code=503, detail="Store not available")
        title = payload.get("title", "New Chat") if payload else "New Chat"
        conv = await conv_store.create_conversation(title=title)
        await conv_store.set_active_conversation(conv["id"])
        return conv

    @app.get("/conversations/{conv_id}")
    async def get_conversation(conv_id: str) -> dict[str, Any]:
        """Get full conversation with messages."""
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if not conv_store:
            raise HTTPException(status_code=503, detail="Store not available")
        conv = await conv_store.get_conversation(conv_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    @app.delete("/conversations/{conv_id}")
    async def delete_conversation(conv_id: str) -> dict[str, Any]:
        """Delete a conversation."""
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if not conv_store:
            raise HTTPException(status_code=503, detail="Store not available")
        deleted = await conv_store.delete_conversation(conv_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted", "id": conv_id}

    @app.put("/conversations/{conv_id}")
    async def rename_conv(conv_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Rename a conversation."""
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if not conv_store:
            raise HTTPException(status_code=503, detail="Store not available")
        title = payload.get("title", "")
        if not title:
            raise HTTPException(status_code=400, detail="Title is required")
        ok = await conv_store.rename_conversation(conv_id, title)
        if not ok:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "renamed", "id": conv_id, "title": title}

    @app.post("/conversations/{conv_id}/active")
    async def set_active_conv(conv_id: str) -> dict[str, Any]:
        """Set the active conversation."""
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")
        if not conv_store:
            raise HTTPException(status_code=503, detail="Store not available")
        await conv_store.set_active_conversation(conv_id)
        return {"status": "ok", "active_id": conv_id}

    # Open task result as a new conversation
    @app.post("/tasks/{task_id}/to-conversation")
    async def task_to_conversation(task_id: str) -> dict[str, Any]:
        """Create a new conversation from a completed task's description and result.

        Adds the task description as a user message and the result as an
        assistant message so the user can continue the dialogue.

        Args:
            task_id: Task ID whose result should be opened in chat.

        Returns:
            New conversation dict with id, title, and messages.

        Raises:
            HTTPException: If task not found, store unavailable, or task has no result.
        """
        task_manager = _app_state.get("task_manager")
        conv_store: Optional[ConversationStore] = _app_state.get("conversation_store")

        if not task_manager:
            raise HTTPException(status_code=503, detail="Service not available")
        if not conv_store:
            raise HTTPException(status_code=503, detail="Conversation store not available")

        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        description = task.get("description", "")
        result_data = task.get("result")
        result_text = None
        if result_data and isinstance(result_data, dict):
            result_text = (
                result_data.get("response")
                or result_data.get("summary")
                or result_data.get("status")
            )
        error_text = task.get("error")
        agent_id = task.get("assigned_agent") or "system"

        # Use result or error as the assistant reply
        reply = result_text or error_text
        if not reply:
            raise HTTPException(status_code=400, detail="Task has no result or error to display")

        # Create conversation with a title derived from the task description
        title = description[:50].strip()
        if len(description) > 50:
            title += "..."
        conv = await conv_store.create_conversation(title=title)
        conv_id = conv["id"]

        # Add the task description as a user message
        await conv_store.add_message(conv_id, "user", description)
        # Add the result as an assistant message
        await conv_store.add_message(conv_id, "assistant", reply, agent=agent_id)

        # Set as active conversation
        await conv_store.set_active_conversation(conv_id)

        # Return the full conversation
        full_conv = await conv_store.get_conversation(conv_id)
        return full_conv

    # Cancel specific task endpoint
    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task_endpoint(task_id: str) -> dict[str, Any]:
        """Cancel a specific task (works for pending, in-progress, and chat tasks).

        Args:
            task_id: Task ID to cancel.

        Returns:
            Dictionary confirming cancellation.

        Raises:
            HTTPException: If task not found or cannot be cancelled.
        """
        task_manager = _app_state.get("task_manager")
        if not task_manager:
            raise HTTPException(status_code=503, detail="Service not available")

        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Cancel the running asyncio.Task if it exists
        active = _app_state.get("active_tasks", {})
        if task_id in active:
            active[task_id].cancel()
            active.pop(task_id, None)

        # Update status in task manager
        try:
            await task_manager.cancel_task(task_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"status": "cancelled", "task_id": task_id}

    # ── User preferences (persisted to .agentarmy/prefs.json) ──────

    def _prefs_path() -> Optional[pathlib.Path]:
        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            return None
        return pathlib.Path(project_dir) / ".agentarmy" / "prefs.json"

    def _load_prefs() -> dict[str, Any]:
        path = _prefs_path()
        if not path or not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_prefs(prefs: dict[str, Any]) -> None:
        path = _prefs_path()
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")
        except (IOError, PermissionError):
            pass

    @app.get("/prefs")
    async def get_prefs() -> dict[str, Any]:
        """Get all user preferences."""
        return _load_prefs()

    @app.put("/prefs")
    async def set_prefs(request: Request) -> dict[str, Any]:
        """Update user preferences (merges with existing).

        Body: {"key": "value", ...}
        """
        payload = await request.json()
        prefs = _load_prefs()
        prefs.update(payload)
        _save_prefs(prefs)
        return prefs

    # ── Memory endpoints ──────────────────────────────────────────────

    @app.get("/memory")
    async def get_memories(
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Retrieve stored memories with optional filters.

        Args:
            query: Free-text search across content and tags.
            memory_type: Filter by type (decision, learning, etc.)
            scope: Filter by scope (project, api, etc.)
            limit: Maximum results.

        Returns:
            Dictionary with memories list and stats.
        """
        mem_store: Optional[AgentMemoryStore] = _app_state.get("memory_store")
        if not mem_store:
            return {"memories": [], "stats": {"total": 0, "by_type": {}, "by_scope": {}}}

        memories = mem_store.recall(
            query=query,
            memory_type=memory_type,
            scope=scope,
            limit=limit,
        )
        return {
            "memories": [m.to_dict() for m in memories],
            "stats": mem_store.get_stats(),
            "types": sorted(MEMORY_TYPES),
            "scopes": sorted(MEMORY_SCOPES),
        }

    @app.post("/memory")
    async def create_memory(payload: dict[str, Any]) -> dict[str, Any]:
        """Store a new memory.

        Args:
            payload: {"content": "...", "type": "...", "scope": "...", "tags": [...]}

        Returns:
            The created memory entry.
        """
        mem_store: Optional[AgentMemoryStore] = _app_state.get("memory_store")
        if not mem_store:
            raise HTTPException(status_code=503, detail="Memory store not initialized")

        content = payload.get("content", "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="Content is required")

        entry = mem_store.remember(
            content=content,
            memory_type=payload.get("type", "context"),
            scope=payload.get("scope", "project"),
            tags=payload.get("tags", []),
            created_by=payload.get("created_by", "user"),
        )
        return {"memory": entry.to_dict(), "stats": mem_store.get_stats()}

    @app.delete("/memory/{entry_id}")
    async def delete_memory(entry_id: str) -> dict[str, Any]:
        """Delete a memory by ID.

        Args:
            entry_id: Memory entry ID.

        Returns:
            Confirmation dict.
        """
        mem_store: Optional[AgentMemoryStore] = _app_state.get("memory_store")
        if not mem_store:
            raise HTTPException(status_code=503, detail="Memory store not initialized")

        deleted = mem_store.forget(entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")

        return {"status": "deleted", "id": entry_id, "stats": mem_store.get_stats()}

    # ── Code Reranker endpoint ────────────────────────────────────────

    @app.post("/rerank")
    async def rerank_code(request: Request) -> dict[str, Any]:
        """Rerank code fragment candidates by relevance to a query.

        Uses an LLM-based reranker with BM25 fallback, batching, and caching.

        Body: {
            "query": "Where is the email validation implemented?",
            "candidates": [
                {"id": "src/user/validation.ts:10-30", "content": "...", "metadata": {...}},
                ...
            ],
            "k": 32  (optional, default 32)
        }

        Returns:
            Ranked results with scores, source info, and config.
        """
        from src.core.reranker import CodeReranker, RerankCandidate

        payload = await request.json()
        query = payload.get("query", "").strip()
        raw_candidates = payload.get("candidates", [])
        k = payload.get("k", 32)

        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        if not raw_candidates:
            raise HTTPException(status_code=400, detail="candidates list is required")

        redis = _app_state.get("redis")
        reranker = CodeReranker(k_rerank=k, global_cache=redis)

        candidates = [
            RerankCandidate(
                id=c.get("id", f"candidate-{i}"),
                content=c.get("content", ""),
                metadata=c.get("metadata", {}),
            )
            for i, c in enumerate(raw_candidates)
        ]

        results = await reranker.rerank(query, candidates)
        return {
            "results": [r.to_dict() for r in results],
            "total_candidates": len(raw_candidates),
            "returned": len(results),
            "query": query,
            "config": reranker.get_config(),
        }

    # ── Figma Design Token extraction endpoints ────────────────────────

    @app.post("/figma/tokens")
    async def extract_figma_tokens(request: Request) -> dict[str, Any]:
        """Extract design tokens from a Figma file.

        Body: {
            "file_key": "abc123",
            "node_id": "1:23" (optional — scope to a frame),
            "format": "json" | "css" | "tailwind" (default: json)
        }

        Returns:
            Design tokens in requested format.
        """
        payload = await request.json()
        file_key = payload.get("file_key", "").strip()
        node_id = payload.get("node_id")
        output_format = payload.get("format", "json").lower()

        if not file_key:
            raise HTTPException(status_code=400, detail="file_key is required")

        figma: FigmaBridge = _app_state.get("figma_bridge") or FigmaBridge()
        if not figma.is_configured:
            raise HTTPException(
                status_code=400,
                detail="Figma not configured. Add your access token in Settings.",
            )

        token_data = await figma.extract_design_tokens(file_key, node_id)

        if token_data.get("error"):
            raise HTTPException(status_code=400, detail=token_data["error"])

        if output_format == "css":
            css = figma.tokens_to_css(token_data)
            return {"format": "css", "file_key": file_key, "content": css, "tokens": figma.tokens_to_json(token_data)}
        elif output_format == "tailwind":
            tw = figma.tokens_to_tailwind(token_data)
            return {"format": "tailwind", "file_key": file_key, "content": tw, "tokens": figma.tokens_to_json(token_data)}
        else:
            return {"format": "json", "file_key": file_key, "tokens": figma.tokens_to_json(token_data)}

    @app.post("/figma/tokens/save")
    async def save_figma_tokens(request: Request) -> dict[str, Any]:
        """Extract tokens and save to a file in the project directory.

        Body: {
            "file_key": "abc123",
            "node_id": "1:23" (optional),
            "format": "css" | "tailwind" | "json",
            "output_path": "src/styles/tokens.css" (optional, auto-generated if omitted)
        }

        Returns:
            Confirmation with written file path.
        """
        payload = await request.json()
        file_key = payload.get("file_key", "").strip()
        node_id = payload.get("node_id")
        output_format = payload.get("format", "css").lower()
        output_path = payload.get("output_path", "").strip()

        if not file_key:
            raise HTTPException(status_code=400, detail="file_key is required")

        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            raise HTTPException(status_code=400, detail="No project directory configured.")

        figma: FigmaBridge = _app_state.get("figma_bridge") or FigmaBridge()
        if not figma.is_configured:
            raise HTTPException(status_code=400, detail="Figma not configured.")

        token_data = await figma.extract_design_tokens(file_key, node_id)
        if token_data.get("error"):
            raise HTTPException(status_code=400, detail=token_data["error"])

        # Generate content
        ext_map = {"css": ".css", "tailwind": ".js", "json": ".json"}
        ext = ext_map.get(output_format, ".json")

        if not output_path:
            output_path = f"src/styles/figma-tokens{ext}"

        if output_format == "css":
            content = figma.tokens_to_css(token_data)
        elif output_format == "tailwind":
            content = figma.tokens_to_tailwind(token_data)
        else:
            import json as json_mod
            content = json_mod.dumps(figma.tokens_to_json(token_data), indent=2)

        # Write to project
        root = pathlib.Path(project_dir).resolve()
        target = (root / output_path).resolve()

        if not str(target).startswith(str(root)):
            raise HTTPException(status_code=403, detail="Path escapes project directory")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        await logger.ainfo("figma_tokens_saved", path=str(target), format=output_format)

        return {
            "status": "saved",
            "path": str(target),
            "relative": output_path,
            "format": output_format,
            "token_summary": {
                "colors": len(token_data.get("tokens", {}).get("colors", {})),
                "typography": len(token_data.get("tokens", {}).get("typography", {})),
                "effects": len(token_data.get("tokens", {}).get("effects", [])),
                "radii": len(token_data.get("tokens", {}).get("radii", [])),
                "spacing": len(token_data.get("tokens", {}).get("spacing", [])),
            },
        }

    # ── Image Generation (Gemini Nano Banana) ───────────────────────
    @app.post("/api/generate-image")
    async def generate_image(request: Request) -> dict[str, Any]:
        """Generate an image using Gemini's Nano Banana model.

        Body: {"prompt": "A futuristic cityscape at sunset", "aspect_ratio": "16:9"}

        Returns:
            Dict with url to the generated image.
        """
        import base64
        import hashlib
        import os
        import time

        body = await request.json()
        prompt = body.get("prompt", "").strip()
        aspect_ratio = body.get("aspect_ratio", "1:1")
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        # Valid aspect ratios for Gemini image generation
        valid_ratios = {"1:1", "3:4", "4:3", "9:16", "16:9"}
        if aspect_ratio not in valid_ratios:
            aspect_ratio = "1:1"

        gemini_key = os.environ.get("AGENTARMY_GEMINI_API_KEY", "")
        if not gemini_key:
            raise HTTPException(
                status_code=400,
                detail="Gemini API key not configured. Add it in Settings > LLM Providers.",
            )

        try:
            import httpx as _httpx

            # Use Gemini REST API directly (avoids heavy SDK dependency)
            # gemini-2.0-flash-exp was deprecated; use gemini-2.5-flash-image
            api_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash-image:generateContent?key={gemini_key}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                },
            }

            async with _httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(api_url, json=payload)

            if resp.status_code != 200:
                err_detail = resp.text[:500]
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Gemini API error: {err_detail}",
                )

            data = resp.json()

            # Extract inline image from response parts
            image_data = None
            mime_type = "image/png"
            text_response = ""
            candidates = data.get("candidates", [])
            for candidate in candidates:
                for part in candidate.get("content", {}).get("parts", []):
                    if "inlineData" in part:
                        image_data = part["inlineData"].get("data")
                        mime_type = part["inlineData"].get("mimeType", "image/png")
                    elif "text" in part:
                        text_response += part["text"]

            if not image_data:
                return {
                    "success": False,
                    "error": "No image generated. " + (text_response or "Try a different prompt."),
                }

            # Save to generated/ directory
            ext = "png" if "png" in mime_type else "jpg" if "jpeg" in mime_type else "webp"
            img_bytes = base64.b64decode(image_data)
            filename = f"img_{int(time.time())}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}.{ext}"
            out_path = _generated_dir / filename
            out_path.write_bytes(img_bytes)

            img_url = f"/generated/{filename}"
            await logger.ainfo("image_generated", prompt=prompt[:80], file=filename, size=len(img_bytes))

            return {
                "success": True,
                "url": img_url,
                "filename": filename,
                "prompt": prompt,
                "text": text_response,
                "size_bytes": len(img_bytes),
            }

        except HTTPException:
            raise
        except Exception as exc:
            await logger.aerror("image_generation_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}")

    # ── Shell / Run endpoint ─────────────────────────────────────────
    # Blocked shell patterns for safety
    _BLOCKED_COMMANDS = {
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){", "fork bomb",
        "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
        "chmod -R 777 /", "chown -R", "> /dev/sda",
    }

    @app.post("/run")
    async def run_command(request: Request) -> dict[str, Any]:
        """Execute a shell command in the project directory.

        Body: {"command": "git status"}

        Safety constraints:
        - Only runs within the configured project directory.
        - Times out after 60 seconds.
        - Blocks obviously destructive commands.

        Returns:
            Dict with stdout, stderr, exit_code, success.
        """
        import shlex

        payload = await request.json()
        command = payload.get("command", "").strip()

        if not command:
            raise HTTPException(status_code=400, detail="Missing command")

        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            raise HTTPException(
                status_code=400,
                detail="No project directory configured. Set it in the Project bar first.",
            )

        root = pathlib.Path(project_dir)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail="Project directory does not exist.")

        # Safety: block obviously destructive commands
        cmd_lower = command.lower().strip()
        for blocked in _BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                raise HTTPException(
                    status_code=403,
                    detail=f"Command blocked for safety: contains '{blocked}'",
                )

        await logger.ainfo("run_command", command=command[:200], cwd=project_dir)

        # Build env with integration tokens so shell commands can use them
        import os as _os
        run_env = _os.environ.copy()
        api_port = _os.environ.get("AGENTARMY_PORT", "8000")
        run_env["AGENTARMY_API_URL"] = f"http://localhost:{api_port}"
        figma_b: FigmaBridge | None = _app_state.get("figma_bridge")
        if figma_b and figma_b.is_configured:
            run_env["FIGMA_TOKEN"] = figma_b.access_token
        gh_b: GitHubBridge | None = _app_state.get("github_bridge")
        if gh_b and gh_b.is_configured:
            run_env["GITHUB_TOKEN"] = gh_b.token
        jira_b: JiraBridge | None = _app_state.get("jira_bridge")
        if jira_b and jira_b.is_configured:
            run_env["JIRA_API_TOKEN"] = jira_b.api_token
            run_env["JIRA_EMAIL"] = jira_b.email
            run_env["JIRA_DOMAIN"] = jira_b.domain

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(root),
                env=run_env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            success = proc.returncode == 0

            return {
                "status": "completed",
                "success": success,
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[-4000:] if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace")[-2000:] if stderr else "",
                "command": command,
            }
        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "Command timed out after 60 seconds.",
                "command": command,
            }
        except Exception as exc:
            return {
                "status": "error",
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(exc),
                "command": command,
            }

    # ── Playwright / E2E testing endpoints ────────────────────────────

    @app.get("/test/e2e/status")
    async def playwright_status() -> dict[str, Any]:
        """Check Playwright installation and test configuration.

        Returns:
            Dict with setup info, installed status, test files found.
        """
        project_dir = _app_state.get("project_dir", "")
        if not project_dir:
            return {"installed": False, "message": "No project directory configured"}

        setup = detect_playwright_setup(project_dir)
        runner: Optional[PlaywrightRunner] = _app_state.get("playwright_runner")
        setup["chromium_ready"] = runner.chromium_ready if runner else False
        return setup

    @app.post("/test/e2e/run")
    async def run_e2e_tests(request: Request) -> dict[str, Any]:
        """Run Playwright E2E tests.

        Body: {
            "test_file": "tests/login.spec.ts" (optional),
            "grep": "should login" (optional),
            "headed": false (optional),
            "extra_args": [] (optional)
        }

        Returns:
            Dict with pass/fail status, results summary, stdout/stderr.
        """
        payload = await request.json()
        runner: Optional[PlaywrightRunner] = _app_state.get("playwright_runner")
        if not runner:
            raise HTTPException(status_code=503, detail="Playwright runner not initialized")

        project_dir = _app_state.get("project_dir", "")
        if project_dir:
            runner.set_project_dir(project_dir)

        result = await runner.run_tests(
            test_file=payload.get("test_file"),
            headed=payload.get("headed", False),
            grep=payload.get("grep"),
            extra_args=payload.get("extra_args"),
        )
        return result

    @app.post("/test/e2e/smoke")
    async def smoke_test(request: Request) -> dict[str, Any]:
        """Run a quick smoke test against a URL.

        Body: {
            "url": "http://localhost:3000",
            "checks": ["status", "title", "no_console_errors", ".my-selector"],
            "screenshot": true
        }

        Returns:
            Dict with check results, performance metrics, screenshot path.
        """
        payload = await request.json()
        url = payload.get("url", "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        runner: Optional[PlaywrightRunner] = _app_state.get("playwright_runner")
        if not runner:
            raise HTTPException(status_code=503, detail="Playwright runner not initialized")

        project_dir = _app_state.get("project_dir", "")
        if project_dir:
            runner.set_project_dir(project_dir)

        result = await runner.smoke_test(
            url=url,
            checks=payload.get("checks"),
            screenshot=payload.get("screenshot", True),
        )
        return result

    @app.post("/test/e2e/scaffold")
    async def scaffold_test(request: Request) -> dict[str, Any]:
        """Generate a Playwright test file scaffold.

        Body: {
            "test_name": "login flow",
            "url": "http://localhost:3000/login",
            "language": "typescript",
            "assertions": ["login button visible", "form submits"]
        }

        Returns:
            Dict with generated file path and content.
        """
        payload = await request.json()
        test_name = payload.get("test_name", "").strip()
        url = payload.get("url", "").strip()

        if not test_name or not url:
            raise HTTPException(status_code=400, detail="test_name and url are required")

        runner: Optional[PlaywrightRunner] = _app_state.get("playwright_runner")
        if not runner:
            raise HTTPException(status_code=503, detail="Playwright runner not initialized")

        project_dir = _app_state.get("project_dir", "")
        if project_dir:
            runner.set_project_dir(project_dir)

        result = runner.scaffold_test(
            test_name=test_name,
            url=url,
            language=payload.get("language", "typescript"),
            assertions=payload.get("assertions"),
        )

        # Optionally write the file to disk
        if payload.get("write", False) and project_dir:
            file_path = pathlib.Path(project_dir) / result["file_path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(result["content"], encoding="utf-8")
            result["written"] = True
            result["full_path"] = str(file_path)

        return result

    @app.post("/test/e2e/install")
    async def install_playwright(request: Request) -> dict[str, Any]:
        """Install Playwright in the project.

        Body: {"language": "auto" | "typescript" | "python"}

        Returns:
            Dict with installation result.
        """
        payload = await request.json()
        runner: Optional[PlaywrightRunner] = _app_state.get("playwright_runner")
        if not runner:
            raise HTTPException(status_code=503, detail="Playwright runner not initialized")

        project_dir = _app_state.get("project_dir", "")
        if project_dir:
            runner.set_project_dir(project_dir)

        result = await runner.install(language=payload.get("language", "auto"))
        return result

    # ── Google Calendar proxy endpoints ─────────────────────────────
    # These proxy through to the Google Calendar API so agents can access
    # calendar data via simple internal HTTP calls.

    @app.get("/api/calendar/calendars")
    async def list_google_calendars() -> dict[str, Any]:
        """List all Google Calendar calendars.

        Returns:
            Dict with calendar list.
        """
        import os
        import httpx as _httpx

        token = os.environ.get("AGENTARMY_GOOGLE_CALENDAR_TOKEN", "")
        if not token:
            return {"calendars": [], "error": "Google Calendar not configured. Add token in Settings."}

        try:
            async with _httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://www.googleapis.com/calendar/v3/users/me/calendarList",
                    headers={"Authorization": f"Bearer {token}"},
                )
            if resp.status_code != 200:
                return {"calendars": [], "error": f"Google API error: {resp.status_code}"}

            data = resp.json()
            calendars = [
                {
                    "id": item.get("id"),
                    "summary": item.get("summary"),
                    "primary": item.get("primary", False),
                    "backgroundColor": item.get("backgroundColor"),
                }
                for item in data.get("items", [])
            ]
            return {"calendars": calendars}
        except Exception as exc:
            return {"calendars": [], "error": str(exc)}

    @app.get("/api/calendar/events")
    async def list_google_events(
        calendar_id: str = "primary",
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 50,
        query: Optional[str] = None,
    ) -> dict[str, Any]:
        """List events from a Google Calendar.

        Args:
            calendar_id: Calendar ID (default: primary).
            time_min: Start of range (ISO 8601).
            time_max: End of range (ISO 8601).
            max_results: Max events to return.
            query: Free text search filter.

        Returns:
            Dict with events list.
        """
        import os
        import httpx as _httpx
        from urllib.parse import quote

        token = os.environ.get("AGENTARMY_GOOGLE_CALENDAR_TOKEN", "")
        if not token:
            return {"events": [], "error": "Google Calendar not configured."}

        params: dict[str, Any] = {
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query

        try:
            url = f"https://www.googleapis.com/calendar/v3/calendars/{quote(calendar_id, safe='')}/events"
            async with _httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    url,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
            if resp.status_code != 200:
                return {"events": [], "error": f"Google API error: {resp.status_code}"}

            data = resp.json()
            events = [
                {
                    "id": item.get("id"),
                    "summary": item.get("summary", "(No title)"),
                    "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                    "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                    "location": item.get("location"),
                    "description": item.get("description", "")[:200],
                    "attendees": [
                        a.get("email") for a in item.get("attendees", [])
                    ][:10],
                    "status": item.get("status"),
                    "htmlLink": item.get("htmlLink"),
                }
                for item in data.get("items", [])
            ]
            return {"events": events, "calendar_id": calendar_id}
        except Exception as exc:
            return {"events": [], "error": str(exc)}

    @app.post("/api/calendar/free-time")
    async def find_google_free_time(request: Request) -> dict[str, Any]:
        """Find free time across Google Calendars.

        Body: {
            "calendar_ids": ["primary"],
            "time_min": "2025-01-06T08:00:00Z",
            "time_max": "2025-01-10T18:00:00Z"
        }

        Returns:
            Dict with busy times and calculated free slots.
        """
        import os
        import httpx as _httpx

        token = os.environ.get("AGENTARMY_GOOGLE_CALENDAR_TOKEN", "")
        if not token:
            return {"free_slots": [], "error": "Google Calendar not configured."}

        payload = await request.json()
        calendar_ids = payload.get("calendar_ids", ["primary"])
        time_min = payload.get("time_min", "")
        time_max = payload.get("time_max", "")

        if not time_min or not time_max:
            raise HTTPException(status_code=400, detail="time_min and time_max are required")

        try:
            body = {
                "timeMin": time_min,
                "timeMax": time_max,
                "items": [{"id": cid} for cid in calendar_ids],
            }
            async with _httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://www.googleapis.com/calendar/v3/freeBusy",
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                )
            if resp.status_code != 200:
                return {"free_slots": [], "error": f"Google API error: {resp.status_code}"}

            data = resp.json()
            return {
                "calendars": data.get("calendars", {}),
                "time_min": time_min,
                "time_max": time_max,
            }
        except Exception as exc:
            return {"free_slots": [], "error": str(exc)}

    # ── Agent detail endpoints ────────────────────────────────
    def _find_agent(agent_id: str):
        """Resolve an agent by exact or prefix match."""
        agents = _app_state.get("agents", {})
        if agent_id in agents:
            return agents[agent_id]
        for aid, ag in agents.items():
            if aid.startswith(agent_id):
                return ag
        return None

    @app.get("/agents/{agent_id}/detail")
    async def get_agent_detail(agent_id: str) -> dict[str, Any]:
        """Get comprehensive details for a single agent."""
        agent = _find_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        heartbeat = agent.report_status()

        # Load YAML config (best-effort)
        config_yaml = ""
        prefix = agent_id.split("-")[0] if "-" in agent_id else agent_id
        config_path = pathlib.Path("config/agents") / f"{prefix}.yaml"
        if config_path.exists():
            try:
                config_yaml = config_path.read_text(encoding="utf-8")
            except Exception:
                config_yaml = ""

        # System context
        try:
            system_context = agent.get_system_context_public()
        except Exception:
            system_context = ""

        return {
            "identity": {
                "id": agent.identity.id,
                "name": agent.identity.name,
                "role": agent.identity.role,
                "security_level": getattr(agent.identity, "security_level", 1),
                "capabilities": [
                    {
                        "name": cap.name,
                        "version": cap.version,
                        "description": cap.description,
                        "parameters": getattr(cap, "parameters", {}),
                    }
                    for cap in agent.identity.capabilities
                ],
            },
            "state": heartbeat.state.value,
            "uptime_seconds": heartbeat.uptime_seconds,
            "tasks_completed": heartbeat.tasks_completed,
            "tasks_failed": heartbeat.tasks_failed,
            "system_context": system_context,
            "config_yaml": config_yaml,
            "errors": agent.get_error_log(limit=20),
        }

    @app.get("/agents/{agent_id}/errors")
    async def get_agent_errors(agent_id: str, limit: int = 50) -> dict[str, Any]:
        """Get agent error log."""
        agent = _find_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        return {"agent_id": agent.identity.id, "errors": agent.get_error_log(limit=limit)}

    @app.put("/agents/{agent_id}/config")
    async def update_agent_config(agent_id: str, request: Request) -> dict[str, Any]:
        """Update agent YAML configuration."""
        import yaml as _yaml

        agent = _find_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        body = await request.json()
        config_text = body.get("config_yaml", "")

        # Validate YAML syntax
        try:
            _yaml.safe_load(config_text)
        except _yaml.YAMLError as exc:
            return {"success": False, "error": f"Invalid YAML: {exc}"}

        prefix = agent_id.split("-")[0] if "-" in agent_id else agent_id
        config_path = pathlib.Path("config/agents") / f"{prefix}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_path.write_text(config_text, encoding="utf-8")
        except Exception as exc:
            return {"success": False, "error": f"Write failed: {exc}"}

        return {"success": True}

    return app


def _build_integrations_context() -> str:
    """Build a context string describing all configured integrations.

    This is injected into the agent's system prompt so it knows which
    tokens / bridges are available and which built-in endpoints to use
    instead of suggesting raw API commands with placeholder tokens.
    """
    import os

    api_port = os.getenv("AGENTARMY_PORT", "8000")
    api_base = f"http://localhost:{api_port}"

    lines: list[str] = []

    # ── Figma ────────────────────────────────────────────────────
    figma: FigmaBridge | None = _app_state.get("figma_bridge")
    if figma and figma.is_configured:
        lines.append(
            "- **Figma** — connected (token configured). "
            "The `$FIGMA_TOKEN` environment variable is available in all bash commands.\n"
            "  **Built-in design-token extraction endpoints:**\n"
            f"  - `curl -s -X POST {api_base}/figma/tokens "
            '-H "Content-Type: application/json" '
            "-d '{\"file_key\": \"<KEY>\", \"node_id\": \"<NODE_ID>\", \"format\": \"css\"}'` "
            "— extracts colors, typography, spacing, shadows, radii as CSS/Tailwind/JSON.\n"
            f"  - `curl -s -X POST {api_base}/figma/tokens/save "
            '-H "Content-Type: application/json" '
            "-d '{\"file_key\": \"<KEY>\", \"format\": \"css\"}'` "
            "— extracts and saves tokens directly to the project.\n"
            "  For raw Figma API calls, use `$FIGMA_TOKEN`: "
            '`curl -s -H "X-Figma-Token: $FIGMA_TOKEN" "https://api.figma.com/v1/..."`\n'
            "  NEVER use placeholder tokens like `<YOUR_FIGMA_TOKEN>` — the real token is always in `$FIGMA_TOKEN`.\n"
            "  **IMPORTANT**: When presenting Figma design data to the user, ALWAYS format it as a clean, "
            "readable summary — organize by category (Colors, Typography, Spacing, Shadows, etc.) with "
            "visual formatting (hex values, font specs, pixel values). NEVER dump raw JSON."
        )

    # ── GitHub ───────────────────────────────────────────────────
    gh_bridge: GitHubBridge | None = _app_state.get("github_bridge")
    if gh_bridge and gh_bridge.is_configured:
        lines.append(
            "- **GitHub** — connected (token configured). "
            "The `$GITHUB_TOKEN` environment variable is available in all bash commands. "
            "Use `git` commands directly — authenticated cloning and API calls are handled. "
            "For GitHub API calls: "
            '`curl -s -H "Authorization: Bearer $GITHUB_TOKEN" "https://api.github.com/..."`. '
            "NEVER suggest placeholder tokens like `<YOUR_GITHUB_TOKEN>`."
        )

    # ── Jira ─────────────────────────────────────────────────────
    jira_bridge: JiraBridge | None = _app_state.get("jira_bridge")
    if jira_bridge and jira_bridge.is_configured:
        domain = os.getenv("AGENTARMY_JIRA_DOMAIN", "")
        lines.append(
            f"- **Jira** — connected (email + API token configured, domain: {domain}). "
            "The `$JIRA_API_TOKEN`, `$JIRA_EMAIL`, and `$JIRA_DOMAIN` environment variables are available. "
            "Jira issues linked in messages are automatically fetched and included as context. "
            "NEVER suggest `curl` commands with placeholder Jira credentials."
        )

    # ── Gemini Image Generation ────────────────────────────────────
    gemini_key = os.getenv("AGENTARMY_GEMINI_API_KEY", "")
    if gemini_key:
        lines.append(
            "- **Image Generation (Gemini Nano Banana)** — available. "
            "You can generate images via the built-in API endpoint:\n"
            f"  `curl -s -X POST {api_base}/api/generate-image "
            '-H "Content-Type: application/json" '
            "-d '{\"prompt\": \"describe the image\", \"aspect_ratio\": \"16:9\"}'`\n"
            "  Aspect ratios: 1:1, 3:4, 4:3, 9:16, 16:9. "
            "The response returns `{\"success\": true, \"url\": \"/generated/img_xxx.png\"}`. "
            "To display the image in your response, use markdown: `![description](URL)`.\n"
            "  Users can also use the `/imagine` command in the chat to generate images directly."
        )

    if not lines:
        return ""

    return (
        "## CONFIGURED INTEGRATIONS\n"
        "The following third-party integrations are set up and authenticated. "
        "All integration tokens are injected as environment variables in bash commands. "
        "NEVER use placeholder tokens — real credentials are always available.\n\n"
        + "\n".join(lines)
    )


def _route_chat_to_agent(message: str, agents: dict) -> Optional[Any]:
    """Route a chat message to the best agent based on keyword matching.

    Analyzes the message text and picks the most relevant agent.
    Falls back to scout for general/unknown queries.

    Args:
        message: The user's chat message.
        agents: Dictionary of agent_id → agent instance.

    Returns:
        The best matching agent, or None.
    """
    msg = message.lower()

    routing_keywords = {
        "sentinel": [
            "security", "vulnerability", "scan", "threat", "attack",
            "cve", "exploit", "penetration", "firewall", "malware",
            "audit", "compliance", "risk",
        ],
        "builder": [
            "code", "build", "develop", "implement", "create app",
            "program", "software", "function", "class", "api",
            "feature", "module", "refactor", "debug",
        ],
        "inspector": [
            "test", "quality", "qa", "bug", "inspect", "coverage",
            "unit test", "integration test", "regression",
        ],
        "watcher": [
            "monitor", "health", "status", "uptime", "alert",
            "metric", "performance", "cpu", "memory", "latency",
        ],
        "scout": [
            "research", "analyze", "compare", "evaluate", "study",
            "investigate", "technology", "trend", "report", "summary",
        ],
        "scribe": [
            "document", "write doc", "readme", "api doc",
            "documentation", "changelog", "spec", "wiki",
        ],
        "devops": [
            "deploy", "infrastructure", "ci/cd", "pipeline", "docker",
            "kubernetes", "server", "terraform", "ansible", "cloud",
        ],
        "marketer": [
            "marketing", "linkedin", "post", "social media", "content",
            "campaign", "brand", "seo", "newsletter",
        ],
        "designer": [
            "design image", "generate image", "social image", "banner",
            "thumbnail", "cover image", "profile picture", "og image",
            "instagram image", "youtube thumbnail", "twitter header",
            "linkedin banner", "facebook image", "visual design",
            "image for", "create visual",
        ],
        "linter": [
            "lint", "eslint", "prettier", "ruff", "pylint", "mypy",
            "stylelint", "format code", "code style", "static analysis",
            "auto-fix", "autofix", "linting", "type check", "typecheck",
            "code quality", "style guide", "rubocop", "clippy",
            "shellcheck", "hadolint", "golangci",
        ],
        "automator": [
            "automate", "automation", "shortcut", "shortcuts", "applescript",
            "osascript", "fill form", "scrape", "browser automate", "rpa",
            "repetitive", "clipboard", "desktop",
            "open app", "run shortcut", "macro",
        ],
        "scheduler": [
            "calendar", "schedule", "meeting", "event", "agenda",
            "free time", "availability", "timesheet", "tempo",
            "google calendar", "gcal", "ical", "appointment",
            "book time", "find slot", "busy", "focus time",
            "weekly overview", "daily agenda", "reschedule",
        ],
    }

    best_prefix = None
    best_score = 0

    for prefix, keywords in routing_keywords.items():
        score = sum(1 for kw in keywords if kw in msg)
        if score > best_score:
            best_score = score
            best_prefix = prefix

    # Use keyword match if we found one
    if best_prefix and best_score > 0:
        for agent_id, agent in agents.items():
            if agent_id.startswith(best_prefix):
                return agent

    # Default to scout for general questions
    for agent_id, agent in agents.items():
        if agent_id.startswith("scout"):
            return agent

    return None


# ── Multi-agent routing & parallel dispatch ─────────────────────────

# Shared routing keyword map (used by both single- and multi-agent routing)
_ROUTING_KEYWORDS: dict[str, list[str]] = {
    "sentinel": [
        "security", "vulnerability", "scan", "threat", "attack",
        "cve", "exploit", "penetration", "firewall", "malware",
        "audit", "compliance", "risk",
    ],
    "builder": [
        "code", "build", "develop", "implement", "create app",
        "program", "software", "function", "class", "api",
        "feature", "module", "refactor", "debug",
    ],
    "inspector": [
        "test", "quality", "qa", "bug", "inspect", "coverage",
        "unit test", "integration test", "regression",
    ],
    "watcher": [
        "monitor", "health", "status", "uptime", "alert",
        "metric", "performance", "cpu", "memory", "latency",
    ],
    "scout": [
        "research", "analyze", "compare", "evaluate", "study",
        "investigate", "technology", "trend", "report", "summary",
    ],
    "scribe": [
        "document", "write doc", "readme", "api doc",
        "documentation", "changelog", "spec", "wiki",
    ],
    "devops": [
        "deploy", "infrastructure", "ci/cd", "pipeline", "docker",
        "kubernetes", "server", "terraform", "ansible", "cloud",
    ],
    "marketer": [
        "marketing", "linkedin", "post", "social media", "content",
        "campaign", "brand", "seo", "newsletter",
    ],
    "designer": [
        "design image", "generate image", "social image", "banner",
        "thumbnail", "cover image", "profile picture", "og image",
        "instagram image", "youtube thumbnail", "twitter header",
        "linkedin banner", "facebook image", "visual design",
        "image for", "create visual",
    ],
    "linter": [
        "lint", "eslint", "prettier", "ruff", "pylint", "mypy",
        "stylelint", "format code", "code style", "static analysis",
        "auto-fix", "autofix", "linting", "type check", "typecheck",
        "code quality", "style guide", "rubocop", "clippy",
        "shellcheck", "hadolint", "golangci",
    ],
    "automator": [
        "automate", "automation", "shortcut", "shortcuts", "applescript",
        "osascript", "fill form", "scrape", "browser automate", "rpa",
        "repetitive", "clipboard", "desktop",
        "open app", "run shortcut", "macro",
    ],
    "scheduler": [
        "calendar", "schedule", "meeting", "event", "agenda",
        "free time", "availability", "timesheet", "tempo",
        "google calendar", "gcal", "ical", "appointment",
        "book time", "find slot", "busy", "focus time",
        "weekly overview", "daily agenda", "reschedule",
    ],
}

# Conjunction phrases that suggest a compound, multi-agent request.
# Space-padded so "android" doesn't match " and ".
_CONJUNCTION_PHRASES = [
    " and ", " with a ", " with an ", " also ",
    " as well as ", " plus ", " & ",
]

# Agent labels + emojis for the merged response header
_AGENT_LABEL: dict[str, str] = {
    "sentinel":  "\U0001f6e1\ufe0f Sentinel",
    "builder":   "\U0001f528 Builder",
    "inspector": "\U0001f50d Inspector",
    "watcher":   "\U0001f4c8 Watcher",
    "scout":     "\U0001f9ed Scout",
    "scribe":    "\U0001f4dd Scribe",
    "devops":    "\u2699\ufe0f DevOps",
    "marketer":  "\U0001f4e3 Marketer",
    "designer":  "\U0001f3a8 Designer",
    "linter":    "\u2705 Linter",
    "automator": "\u26a1 Automator",
    "scheduler": "\U0001f4c5 Scheduler",
}

# Conflict pairs — when both appear, keep only the more specific one.
_AGENT_CONFLICTS: list[tuple[str, str]] = [
    ("builder", "linter"),       # linting is a subset → keep linter
    ("designer", "scout"),       # drop generic scout
]

# Specificity score — higher = more specific, wins conflicts
_AGENT_SPECIFICITY: dict[str, int] = {
    "sentinel": 5, "linter": 5, "scheduler": 5, "designer": 5,
    "builder": 4, "inspector": 4, "scribe": 4, "automator": 4,
    "marketer": 4, "devops": 4, "watcher": 4,
    "scout": 1,  # generic fallback
}


def _deduplicate_agents(candidates: list[tuple[str, int]]) -> list[str]:
    """Remove overlapping agents from a candidate list.

    Uses conflict pairs and specificity scores.  Also drops ``scout``
    when any other agent is present (scout is purely fallback).
    """
    selected: dict[str, int] = dict(candidates)

    # Drop scout if we already have other agents
    if "scout" in selected and len(selected) > 1:
        del selected["scout"]

    # Resolve conflict pairs
    for a, b in _AGENT_CONFLICTS:
        if a in selected and b in selected:
            # Keep the more specific one
            if _AGENT_SPECIFICITY.get(a, 3) >= _AGENT_SPECIFICITY.get(b, 3):
                del selected[b]
            else:
                del selected[a]

    # Sort by score descending, cap at 4
    return [
        prefix
        for prefix, _ in sorted(selected.items(), key=lambda x: -x[1])
    ][:4]


def _route_chat_to_agents(message: str, agents: dict) -> list[Any]:
    """Route a chat message to one **or more** agents.

    When the message contains conjunction phrases (``and``, ``plus``, etc.)
    **and** keywords from 2+ distinct agent domains match, we return
    multiple agents so they can be dispatched in parallel.

    Otherwise falls back to single-agent routing.

    Args:
        message: User's chat message.
        agents:  Dictionary of agent_id → agent instance.

    Returns:
        Non-empty list of agent instances (usually 1, up to 4).
    """
    msg = f" {message.lower()} "

    # Check for conjunction phrases
    has_conjunction = any(conj in msg for conj in _CONJUNCTION_PHRASES)

    if has_conjunction:
        # Score every agent
        scores: dict[str, int] = {}
        for prefix, keywords in _ROUTING_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in msg)
            if score > 0:
                scores[prefix] = score

        if len(scores) >= 2:
            deduped = _deduplicate_agents(list(scores.items()))

            if len(deduped) >= 2:
                # Resolve agent instances
                result: list[Any] = []
                for prefix in deduped:
                    for agent_id, agent in agents.items():
                        if agent_id.startswith(prefix):
                            result.append(agent)
                            break
                if len(result) >= 2:
                    return result

    # Fall back to single-agent routing
    single = _route_chat_to_agent(message, agents)
    return [single] if single else []


def _merge_multi_agent_responses(
    responses: list[dict[str, Any]],
) -> str:
    """Merge per-agent responses into a single markdown string.

    Each agent's contribution gets a heading with its emoji + name,
    separated by horizontal rules.
    """
    parts: list[str] = []
    for r in responses:
        agent_id: str = r.get("agent_id", "unknown")
        prefix = agent_id.split("-")[0]
        label = _AGENT_LABEL.get(prefix, prefix.upper())
        body = r.get("response", "")
        parts.append(f"## {label}\n\n{body}")
    return "\n\n---\n\n".join(parts)


async def _handle_multi_agent_chat(
    message: str,
    agents_list: list[Any],
    task_manager: "TaskManager",
    task_payload: dict[str, Any],
    conv_store: Optional["ConversationStore"],
    conversation_id: Optional[str],
    active_tasks: dict,
) -> dict[str, Any]:
    """Dispatch a single message to multiple agents in parallel.

    Creates an independent task per agent, runs them concurrently via
    ``asyncio.gather``, collects results, and returns a merged response.

    Args:
        message:          Original user message.
        agents_list:      List of agent instances to dispatch to.
        task_manager:     TaskManager for task lifecycle.
        task_payload:     Shared payload (model, project_dir, context, …).
        conv_store:       ConversationStore for persistence (may be None).
        conversation_id:  Active conversation ID (may be None).
        active_tasks:     ``_app_state["active_tasks"]`` dict for cancel support.

    Returns:
        Response dict with ``agent_id="multi-agent"`` and per-agent ``responses``.
    """
    from src.core.task_manager import TaskStatus

    # ── Create one task per agent ──────────────────────────────────
    task_ids: list[str] = []
    async_tasks: list[asyncio.Task] = []
    agent_ids: list[str] = []

    for agent in agents_list:
        tid = await task_manager.create_task(
            description=message,
            priority=3,
            payload={**task_payload, "agent_id": agent.identity.id},
            timeout_seconds=120,
            max_retries=1,
            tags=["chat", "multi-agent"],
        )
        task_ids.append(tid)
        agent_ids.append(agent.identity.id)
        task_obj = task_manager.get_task(tid)

        await task_manager.assign_task(tid, agent.identity.id)
        await task_manager.update_task_status(tid, TaskStatus.IN_PROGRESS)

        processing = asyncio.create_task(agent._process_task_with_audit(task_obj))
        active_tasks[tid] = processing
        async_tasks.append(processing)

    # ── Await all in parallel ──────────────────────────────────────
    raw_results = await asyncio.gather(*async_tasks, return_exceptions=True)

    # ── Collect per-agent responses ────────────────────────────────
    responses: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_results):
        tid = task_ids[idx]
        aid = agent_ids[idx]
        active_tasks.pop(tid, None)

        if isinstance(raw, BaseException):
            # Agent raised or was cancelled
            error_msg = str(raw)[:200] if not isinstance(raw, asyncio.CancelledError) else "Cancelled"
            try:
                await task_manager.fail_task(tid, error_msg)
            except Exception:
                pass
            responses.append({
                "agent_id": aid,
                "response": f"⚠️ {aid} encountered an error: {error_msg}",
                "task_id": tid,
                "success": False,
            })
        else:
            result, success = raw
            if success:
                try:
                    await task_manager.complete_task(tid, result)
                except Exception:
                    pass
                text = (
                    result.get("response")
                    or result.get("summary")
                    or result.get("status", "Task completed.")
                )
            else:
                err = result.get("error", "Processing failed")
                try:
                    await task_manager.fail_task(tid, err)
                except Exception:
                    pass
                text = f"I encountered an issue: {err}"

            responses.append({
                "agent_id": aid,
                "response": text,
                "task_id": tid,
                "success": success,
            })

    # ── Build merged response text ─────────────────────────────────
    merged = _merge_multi_agent_responses(responses)

    # ── Persist to conversation store ──────────────────────────────
    if conv_store and conversation_id:
        await conv_store.add_message(
            conversation_id, "assistant", merged, agent="multi-agent",
        )

    return {
        "response": merged,
        "agent_id": "multi-agent",
        "task_ids": task_ids,
        "agents": agent_ids,
        "responses": responses,
        "cancelled": False,
    }


async def _process_task(
    task_manager: TaskManager,
    agent: Any,
    task: dict[str, Any],
) -> None:
    """Process a task with an agent.

    Args:
        task_manager: Task manager instance.
        agent: Agent to process task.
        task: Task to process.
    """
    try:
        await task_manager.update_task_status(
            task["id"],
            TaskStatus.IN_PROGRESS,
        )

        # Process task
        result, success = await agent._process_task_with_audit(task)

        if success:
            await task_manager.complete_task(task["id"], result)
        else:
            await task_manager.fail_task(task["id"], result.get("error", "Unknown error"))

    except Exception as exc:
        await logger.aerror(
            "task_processing_error",
            task_id=task["id"],
            error=str(exc),
        )
        await task_manager.fail_task(task["id"], str(exc))


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
