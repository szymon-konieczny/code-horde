"""
Setup & Settings API — onboarding wizard backend.

Endpoints:
  GET  /api/setup/status   — which keys are configured (masked)
  POST /api/setup/save     — save keys to .env
  POST /api/setup/test     — test a specific provider connection
  GET  /api/setup/infra    — check Docker infrastructure health
"""

import asyncio
import os
import pathlib
import re
import socket
from typing import Any, Optional

import httpx
import structlog
from fastapi import APIRouter, HTTPException

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])

# ── Helpers ──────────────────────────────────────────────────────────

def _find_env_path() -> pathlib.Path:
    """Locate the .env file (project root)."""
    # Walk up from this file to find project root
    current = pathlib.Path(__file__).resolve()
    for parent in [current.parent.parent.parent, pathlib.Path.cwd()]:
        env_path = parent / ".env"
        if env_path.exists():
            return env_path
    # Default: project root / .env
    return pathlib.Path.cwd() / ".env"


def _read_env(env_path: pathlib.Path) -> dict[str, str]:
    """Parse a .env file into a dict."""
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        result[key] = value
    return result


def _write_env(env_path: pathlib.Path, updates: dict[str, str]) -> None:
    """Update .env file — preserves comments and ordering, adds new keys at end."""
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    # Track which keys we've already written
    written_keys: set[str] = set()
    output_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                # Replace with new value
                output_lines.append(f"{key}={updates[key]}")
                written_keys.add(key)
                continue
        output_lines.append(line)

    # Append any new keys not yet in the file
    for key, value in updates.items():
        if key not in written_keys:
            output_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    from src.platform import get_subprocess_adapter
    get_subprocess_adapter().chmod_safe(env_path, 0o600)


def _mask_key(value: str) -> str:
    """Mask a secret key for display: 'sk-ant-abc123xyz' → 'sk-ant-•••xyz'."""
    if not value or value.startswith("your_") or value.startswith("YOUR_"):
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return value[:4] + "•" * (len(value) - 7) + value[-3:]


def _is_port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """Check if a TCP port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


# ── Key configuration map ────────────────────────────────────────────

PROVIDER_KEYS = {
    "claude": {
        "env_key": "AGENTARMY_CLAUDE_API_KEY",
        "label": "Claude (Anthropic)",
        "required": False,
        "placeholder": "sk-ant-...",
        "test_url": "https://api.anthropic.com/v1/messages",
        "group": "llm",
    },
    "openai": {
        "env_key": "AGENTARMY_OPENAI_API_KEY",
        "label": "OpenAI",
        "required": False,
        "placeholder": "sk-...",
        "test_url": "https://api.openai.com/v1/models",
        "group": "llm",
    },
    "huggingface": {
        "env_key": "AGENTARMY_HF_API_KEY",
        "label": "Hugging Face",
        "required": False,
        "placeholder": "hf_...",
        "test_url": "https://huggingface.co/api/whoami-v2",
        "group": "llm",
    },
    "gemini": {
        "env_key": "AGENTARMY_GEMINI_API_KEY",
        "label": "Google Gemini",
        "required": False,
        "placeholder": "AIza...",
        "test_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "group": "llm",
    },
    "kimi": {
        "env_key": "AGENTARMY_KIMI_API_KEY",
        "label": "Kimi (Moonshot)",
        "required": False,
        "placeholder": "sk-...",
        "test_url": "https://api.moonshot.cn/v1/models",
        "group": "llm",
    },
    "bielik": {
        "env_key": "AGENTARMY_BIELIK_MODEL",
        "label": "Bielik (Polish LLM)",
        "required": False,
        "placeholder": "speakleash/bielik-11b-v3.0-instruct",
        "group": "llm",
    },
    "whatsapp_token": {
        "env_key": "AGENTARMY_WHATSAPP_API_TOKEN",
        "label": "WhatsApp API Token",
        "required": False,
        "placeholder": "EAA...",
        "group": "whatsapp",
    },
    "whatsapp_phone": {
        "env_key": "AGENTARMY_WHATSAPP_PHONE_NUMBER_ID",
        "label": "WhatsApp Phone Number ID",
        "required": False,
        "placeholder": "1234567890",
        "group": "whatsapp",
    },
    "whatsapp_verify": {
        "env_key": "AGENTARMY_WHATSAPP_VERIFY_TOKEN",
        "label": "WhatsApp Verify Token",
        "required": False,
        "placeholder": "any random string",
        "group": "whatsapp",
    },
    "neo4j_password": {
        "env_key": "AGENTARMY_NEO4J_PASSWORD",
        "label": "Neo4j Password",
        "required": False,
        "placeholder": "localdev_neo4j_2026",
        "group": "infra",
    },
    "jira_email": {
        "env_key": "AGENTARMY_JIRA_EMAIL",
        "label": "Jira Email",
        "required": False,
        "placeholder": "you@company.com",
        "group": "integrations",
    },
    "jira_token": {
        "env_key": "AGENTARMY_JIRA_API_TOKEN",
        "label": "Jira API Token",
        "required": False,
        "placeholder": "ATATT3x...",
        "test_url": "https://api.atlassian.com/me",
        "group": "integrations",
    },
    "jira_domain": {
        "env_key": "AGENTARMY_JIRA_DOMAIN",
        "label": "Jira Domain",
        "required": False,
        "placeholder": "mycompany (for mycompany.atlassian.net)",
        "group": "integrations",
    },
    "figma_token": {
        "env_key": "AGENTARMY_FIGMA_TOKEN",
        "label": "Figma Access Token",
        "required": False,
        "placeholder": "figd_...",
        "test_url": "https://api.figma.com/v1/me",
        "group": "integrations",
    },
    "github_token": {
        "env_key": "AGENTARMY_GITHUB_TOKEN",
        "label": "GitHub Personal Access Token",
        "required": False,
        "placeholder": "ghp_...",
        "test_url": "https://api.github.com/user",
        "group": "integrations",
    },
}


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/status")
async def setup_status() -> dict[str, Any]:
    """Return configuration status — which keys are set (masked values)."""
    env_path = _find_env_path()
    env_data = _read_env(env_path)

    providers: dict[str, Any] = {}

    # Track whether at least one LLM provider has a key
    any_llm_configured = False
    LLM_PROVIDER_IDS = {"claude", "openai", "gemini", "kimi", "huggingface"}

    for provider_id, meta in PROVIDER_KEYS.items():
        raw_value = env_data.get(meta["env_key"], "")
        is_set = bool(raw_value) and not raw_value.startswith("your_") and not raw_value.startswith("YOUR_")

        if is_set and provider_id in LLM_PROVIDER_IDS:
            any_llm_configured = True

        providers[provider_id] = {
            "label": meta["label"],
            "env_key": meta["env_key"],
            "is_set": is_set,
            "masked_value": _mask_key(raw_value) if is_set else "",
            "required": meta.get("required", False),
            "placeholder": meta["placeholder"],
            "group": meta["group"],
        }

    return {
        "configured": any_llm_configured,
        "env_file_exists": env_path.exists(),
        "env_file_path": str(env_path),
        "providers": providers,
    }


@router.post("/save")
async def save_keys(payload: dict[str, Any]) -> dict[str, Any]:
    """Save API keys to .env file.

    Body: {"keys": {"claude": "sk-ant-...", "openai": "sk-...", ...}}
    """
    keys = payload.get("keys", {})
    if not keys:
        raise HTTPException(status_code=400, detail="No keys provided")

    env_path = _find_env_path()

    # Map provider IDs to env variable names
    updates: dict[str, str] = {}
    saved: list[str] = []

    for provider_id, value in keys.items():
        value = str(value).strip()
        if not value:
            continue
        meta = PROVIDER_KEYS.get(provider_id)
        if meta:
            updates[meta["env_key"]] = value
            saved.append(meta["label"])
        else:
            # Allow arbitrary AGENTARMY_* keys
            if provider_id.startswith("AGENTARMY_"):
                updates[provider_id] = value
                saved.append(provider_id)

    if not updates:
        raise HTTPException(status_code=400, detail="No valid keys to save")

    # Also ensure .env exists (copy from .env.local if needed)
    if not env_path.exists():
        template = env_path.parent / ".env.local"
        if template.exists():
            import shutil
            shutil.copy2(template, env_path)
            env_path.chmod(0o600)

    _write_env(env_path, updates)

    # Reload into current process environment
    for key, value in updates.items():
        os.environ[key] = value

    await logger.ainfo("setup_keys_saved", providers=saved)

    return {
        "saved": saved,
        "count": len(saved),
    }


@router.post("/test")
async def test_connection(payload: dict[str, Any]) -> dict[str, Any]:
    """Test a provider connection.

    Body: {"provider": "claude", "key": "sk-ant-..." (optional, uses saved)}
    """
    provider_id = payload.get("provider", "")
    explicit_key = payload.get("key", "")

    meta = PROVIDER_KEYS.get(provider_id)
    if not meta:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")

    # Get key: explicit > env > .env file
    api_key = explicit_key or os.environ.get(meta["env_key"], "")
    if not api_key:
        env_data = _read_env(_find_env_path())
        api_key = env_data.get(meta["env_key"], "")

    if not api_key:
        return {"provider": provider_id, "status": "no_key", "message": "No API key configured"}

    # Test based on provider
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if provider_id == "claude":
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                )
                if resp.status_code in (200, 201):
                    return {"provider": provider_id, "status": "ok", "message": "Connected"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid API key"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "openai":
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    return {"provider": provider_id, "status": "ok", "message": "Connected"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid API key"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "huggingface":
                resp = await client.get(
                    "https://huggingface.co/api/whoami-v2",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    name = data.get("name", "unknown")
                    return {"provider": provider_id, "status": "ok", "message": f"Connected as {name}"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid token"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "gemini":
                resp = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": api_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    model_count = len(data.get("models", []))
                    return {"provider": provider_id, "status": "ok", "message": f"Connected ({model_count} models)"}
                elif resp.status_code == 400:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid API key"}
                elif resp.status_code == 403:
                    return {"provider": provider_id, "status": "invalid_key", "message": "API key not authorized"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "kimi":
                resp = await client.get(
                    "https://api.moonshot.cn/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    model_count = len(data.get("data", []))
                    return {"provider": provider_id, "status": "ok", "message": f"Connected ({model_count} models)"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid API key"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "whatsapp_token":
                # Test WhatsApp token against Meta Graph API
                resp = await client.get(
                    "https://graph.facebook.com/v22.0/me",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    name = data.get("name", "unknown")
                    return {"provider": provider_id, "status": "ok", "message": f"Connected as {name}"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid token"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "figma_token":
                resp = await client.get(
                    "https://api.figma.com/v1/me",
                    headers={"X-Figma-Token": api_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    handle = data.get("handle", data.get("email", "unknown"))
                    return {"provider": provider_id, "status": "ok", "message": f"Connected as {handle}"}
                elif resp.status_code == 403:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid token"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "github_token":
                resp = await client.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/vnd.github+json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    login = data.get("login", "unknown")
                    return {"provider": provider_id, "status": "ok", "message": f"Connected as {login}"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid token"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code}"}

            elif provider_id == "jira_token":
                # Jira uses Basic auth: email:token (base64)
                env_data = _read_env(_find_env_path())
                jira_email_val = os.environ.get("AGENTARMY_JIRA_EMAIL", "") or env_data.get("AGENTARMY_JIRA_EMAIL", "")
                jira_domain_val = os.environ.get("AGENTARMY_JIRA_DOMAIN", "") or env_data.get("AGENTARMY_JIRA_DOMAIN", "")
                if not jira_email_val:
                    return {"provider": provider_id, "status": "error", "message": "Set Jira Email first"}
                if not jira_domain_val:
                    return {"provider": provider_id, "status": "error", "message": "Set Jira Domain first"}

                # Sanitise domain — user might type full URL or include .atlassian.net
                domain = jira_domain_val.strip().lower()
                domain = re.sub(r'^https?://', '', domain)          # strip protocol
                domain = domain.split('/')[0]                        # strip path
                domain = re.sub(r'\.atlassian\.net$', '', domain)    # strip suffix
                domain = domain.strip('.')
                if not domain or not re.match(r'^[a-z0-9][a-z0-9\-]*$', domain):
                    return {
                        "provider": provider_id,
                        "status": "error",
                        "message": f"Invalid Jira domain '{jira_domain_val}'. Use just the subdomain, e.g. 'mycompany'",
                    }

                import base64
                jira_url = f"https://{domain}.atlassian.net/rest/api/3/myself"
                creds = base64.b64encode(f"{jira_email_val}:{api_key}".encode()).decode()
                try:
                    resp = await client.get(
                        jira_url,
                        headers={
                            "Authorization": f"Basic {creds}",
                            "Accept": "application/json",
                        },
                    )
                except httpx.ConnectError as exc:
                    return {
                        "provider": provider_id,
                        "status": "unreachable",
                        "message": f"Cannot reach {domain}.atlassian.net — check domain spelling or network ({type(exc).__name__})",
                    }
                except httpx.TimeoutException:
                    return {
                        "provider": provider_id,
                        "status": "unreachable",
                        "message": f"Timeout connecting to {domain}.atlassian.net — check network or try again",
                    }
                if resp.status_code == 200:
                    data = resp.json()
                    name = data.get("displayName", "unknown")
                    return {"provider": provider_id, "status": "ok", "message": f"Connected as {name}"}
                elif resp.status_code == 401:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Invalid email/token combo"}
                elif resp.status_code == 403:
                    return {"provider": provider_id, "status": "invalid_key", "message": "Token valid but lacks API access — check Jira permissions"}
                elif resp.status_code == 404:
                    return {"provider": provider_id, "status": "error", "message": f"Site '{domain}.atlassian.net' not found — check domain"}
                else:
                    return {"provider": provider_id, "status": "error", "message": f"HTTP {resp.status_code} from {domain}.atlassian.net"}

            elif provider_id in ("jira_email", "jira_domain", "whatsapp_phone", "whatsapp_verify"):
                # These are plain values, not API-testable
                if api_key.strip():
                    return {"provider": provider_id, "status": "ok", "message": "Saved"}
                return {"provider": provider_id, "status": "no_key", "message": "Not set"}

            else:
                return {"provider": provider_id, "status": "skipped", "message": "No test available"}

    except httpx.ConnectError as exc:
        test_url = meta.get("test_url", "API")
        return {"provider": provider_id, "status": "unreachable", "message": f"Cannot reach {test_url} — check network ({type(exc).__name__})"}
    except httpx.TimeoutException:
        test_url = meta.get("test_url", "API")
        return {"provider": provider_id, "status": "unreachable", "message": f"Timeout connecting to {test_url} — try again"}
    except Exception as exc:
        return {"provider": provider_id, "status": "error", "message": str(exc)[:200]}


@router.get("/project")
async def get_project_directory() -> dict[str, Any]:
    """Return the current project directory."""
    env_path = _find_env_path()
    env_data = _read_env(env_path)
    project_dir = env_data.get("AGENTARMY_PROJECT_DIR", "") or os.environ.get("AGENTARMY_PROJECT_DIR", "")

    exists = bool(project_dir) and pathlib.Path(project_dir).is_dir()
    # Gather quick stats if dir exists
    file_count = 0
    has_git = False
    has_agents_md = False
    agents_md_name = None
    if exists:
        p = pathlib.Path(project_dir)
        has_git = (p / ".git").is_dir()
        try:
            file_count = sum(1 for f in p.iterdir() if not f.name.startswith("."))
        except OSError:
            pass
        # Check for agent instruction files
        _AGENT_FILES = ["AGENTS.md", "AGENTS.txt", ".agents.md", "AI_RULES.md", "AI.md", "CLAUDE.md", ".cursorrules"]
        for af in _AGENT_FILES:
            if (p / af).is_file():
                has_agents_md = True
                agents_md_name = af
                break

    return {
        "project_dir": project_dir,
        "exists": exists,
        "has_git": has_git,
        "has_agents_md": has_agents_md,
        "agents_md_name": agents_md_name,
        "file_count": file_count,
    }


@router.post("/project")
async def set_project_directory(payload: dict[str, Any]) -> dict[str, Any]:
    """Set the project directory.

    Body: {"path": "/Users/you/Projects/my-app"}
    """
    raw_path = payload.get("path", "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="Path is required")

    project_path = pathlib.Path(raw_path).expanduser().resolve()
    if not project_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {project_path}")

    # Persist to .env
    env_path = _find_env_path()
    _write_env(env_path, {"AGENTARMY_PROJECT_DIR": str(project_path)})
    os.environ["AGENTARMY_PROJECT_DIR"] = str(project_path)

    # Update app state (including memory store project dir)
    try:
        from src.main import _app_state
        _app_state["project_dir"] = str(project_path)
        mem_store = _app_state.get("memory_store")
        if mem_store:
            mem_store.set_project_dir(str(project_path))
        pw_runner = _app_state.get("playwright_runner")
        if pw_runner:
            pw_runner.set_project_dir(str(project_path))
    except Exception:
        pass

    has_git = (project_path / ".git").is_dir()
    file_count = sum(1 for f in project_path.iterdir() if not f.name.startswith("."))

    # Check for agent instruction files
    has_agents_md = False
    agents_md_name = None
    _AGENT_FILES = ["AGENTS.md", "AGENTS.txt", ".agents.md", "AI_RULES.md", "AI.md", "CLAUDE.md", ".cursorrules"]
    for af in _AGENT_FILES:
        if (project_path / af).is_file():
            has_agents_md = True
            agents_md_name = af
            break

    await logger.ainfo("project_directory_set", path=str(project_path), has_git=has_git, has_agents_md=has_agents_md)

    return {
        "project_dir": str(project_path),
        "exists": True,
        "has_git": has_git,
        "has_agents_md": has_agents_md,
        "agents_md_name": agents_md_name,
        "file_count": file_count,
    }


@router.get("/browse")
async def browse_directory(path: str = "") -> dict[str, Any]:
    """Browse directories on the local filesystem for the project selector.

    Query: ?path=/Users/you/Projects  (defaults to home directory)
    Returns directory listing with metadata for building a file browser UI.
    """
    if not path:
        path = str(pathlib.Path.home())

    target = pathlib.Path(path).expanduser().resolve()

    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    # Gather entries
    entries: list[dict[str, Any]] = []
    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            # Skip hidden files/dirs (but allow navigating into them if path given directly)
            if item.name.startswith("."):
                continue
            try:
                is_dir = item.is_dir()
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": is_dir,
                    "has_git": is_dir and (item / ".git").is_dir(),
                })
            except PermissionError:
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    # Compute parent
    parent = str(target.parent) if target != target.parent else None

    return {
        "current": str(target),
        "parent": parent,
        "entries": entries,
        "is_root": target == target.parent,
    }


@router.get("/infra")
async def check_infrastructure() -> dict[str, Any]:
    """Check health of Docker infrastructure services."""
    services = {
        "postgresql":  {"host": "127.0.0.1", "port": 5432, "required": True},
        "redis":       {"host": "127.0.0.1", "port": 6379, "required": False},
        "rabbitmq":    {"host": "127.0.0.1", "port": 5672, "required": False},
        "neo4j":       {"host": "127.0.0.1", "port": 7687, "required": False},
        "ollama":      {"host": "127.0.0.1", "port": 11434, "required": False},
    }

    results: dict[str, Any] = {}
    for name, info in services.items():
        reachable = _is_port_open(info["host"], info["port"])
        results[name] = {
            "host": info["host"],
            "port": info["port"],
            "status": "running" if reachable else "stopped",
            "required": info["required"],
        }

    all_core_up = all(
        results[s]["status"] == "running"
        for s in ["postgresql"]  # Only PostgreSQL is required; Redis uses in-memory fallback
    )

    return {
        "all_core_running": all_core_up,
        "services": results,
    }


@router.post("/infra/start")
async def start_service(body: dict[str, Any]) -> dict[str, Any]:
    """Start an infrastructure service by name."""
    import shutil
    import subprocess

    name = body.get("service", "").lower()
    if name not in ("neo4j", "ollama", "redis", "rabbitmq"):
        raise HTTPException(status_code=400, detail=f"Unknown service: {name}")

    if name == "ollama":
        # Ollama runs natively (not Docker)
        if not shutil.which("ollama"):
            # Auto-install via Homebrew on macOS
            if shutil.which("brew"):
                try:
                    proc = subprocess.run(
                        ["brew", "install", "ollama"],
                        capture_output=True, text=True, timeout=180,
                    )
                    if proc.returncode != 0:
                        return {"ok": False, "error": f"Failed to install Ollama via Homebrew: {proc.stderr.strip()[:200]}"}
                except subprocess.TimeoutExpired:
                    return {"ok": False, "error": "Ollama Homebrew install timed out — try running 'brew install ollama' manually"}
                except Exception as exc:
                    return {"ok": False, "error": f"Ollama install failed: {str(exc)[:200]}"}
            else:
                return {"ok": False, "error": "Ollama not installed and Homebrew not available. Install from https://ollama.com"}
        # Check if already running
        if _is_port_open("127.0.0.1", 11434):
            return {"ok": True, "message": "Ollama already running"}
        # Start ollama serve in background
        from src.platform import get_subprocess_adapter
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **get_subprocess_adapter().create_process_kwargs(isolated=True),
        )
        # Wait up to 5s for it to come up
        for _ in range(10):
            await asyncio.sleep(0.5)
            if _is_port_open("127.0.0.1", 11434):
                return {"ok": True, "message": "Ollama started"}
        return {"ok": False, "error": "Ollama started but not responding yet — try again in a moment"}

    # ── Homebrew-first fallback for neo4j, redis, rabbitmq ───────
    port_map = {"neo4j": 7687, "redis": 6379, "rabbitmq": 5672}
    brew_service_map = {"neo4j": "neo4j", "redis": "redis", "rabbitmq": "rabbitmq-server"}
    port = port_map.get(name, 0)

    # Check if already running
    if port and _is_port_open("127.0.0.1", port):
        return {"ok": True, "message": f"{name} already running on port {port}"}

    # Try Homebrew first (macOS)
    if shutil.which("brew"):
        brew_pkg = brew_service_map.get(name)
        if brew_pkg:
            try:
                # Check if installed
                check = subprocess.run(["brew", "list", brew_pkg], capture_output=True, timeout=10)
                if check.returncode != 0:
                    # Install via Homebrew
                    install = subprocess.run(
                        ["brew", "install", brew_pkg],
                        capture_output=True, text=True, timeout=180,
                    )
                    if install.returncode != 0:
                        pass  # Fall through to Docker
                    else:
                        # Start via brew services
                        subprocess.run(
                            ["brew", "services", "start", brew_pkg],
                            capture_output=True, timeout=15,
                        )
                        # Wait for port
                        if port:
                            for _ in range(12):
                                await asyncio.sleep(0.5)
                                if _is_port_open("127.0.0.1", port):
                                    return {"ok": True, "message": f"{name} installed and started via Homebrew on port {port}"}
                        return {"ok": True, "message": f"{name} installed via Homebrew (may still be initializing)"}
                else:
                    # Already installed — just start
                    subprocess.run(
                        ["brew", "services", "start", brew_pkg],
                        capture_output=True, timeout=15,
                    )
                    if port:
                        for _ in range(12):
                            await asyncio.sleep(0.5)
                            if _is_port_open("127.0.0.1", port):
                                return {"ok": True, "message": f"{name} started via Homebrew on port {port}"}
                    return {"ok": True, "message": f"{name} brew service started (may still be initializing)"}
            except Exception:
                pass  # Fall through to Docker

    # Docker-based fallback
    compose_bin = shutil.which("docker") and "docker compose"
    if not compose_bin:
        return {"ok": False, "error": f"Cannot start {name}: neither Homebrew nor Docker available. Install via 'brew install {brew_service_map.get(name, name)}' or install Docker."}

    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    compose_file = project_root / "docker" / "docker-compose.local.yaml"
    if not compose_file.exists():
        return {"ok": False, "error": "docker-compose.local.yaml not found"}

    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d", name],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()[:200]
            return {"ok": False, "error": f"Docker failed: {stderr}" if stderr else "Failed to start container"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}

    # Wait for port to open
    if port:
        for _ in range(10):
            await asyncio.sleep(0.5)
            if _is_port_open("127.0.0.1", port):
                return {"ok": True, "message": f"{name} started via Docker on port {port}"}

    return {"ok": True, "message": f"{name} container started (may still be initializing)"}
