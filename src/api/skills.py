"""REST API for skill management in Code Horde.

Provides endpoints for skill discovery, CRUD, validation, dependency
resolution, hot-reload, and audit logging.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.core.skills import SkillDefinition, SkillDependency, SkillRegistry

router = APIRouter(prefix="/api/skills", tags=["skills"])

# ---------------------------------------------------------------------------
# Module-level state — wired by main.py at startup
# ---------------------------------------------------------------------------

_registry: Optional[SkillRegistry] = None
_agents: Optional[dict[str, Any]] = None  # agent_id -> BaseAgent


def init_skills_api(registry: SkillRegistry, agents: dict[str, Any]) -> None:
    """Called once from ``main.py`` to inject dependencies."""
    global _registry, _agents
    _registry = registry
    _agents = agents


def _get_registry() -> SkillRegistry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Skill registry not initialised")
    return _registry


def _get_agents() -> dict[str, Any]:
    if _agents is None:
        raise HTTPException(status_code=503, detail="Agents not initialised")
    return _agents


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SkillCreateRequest(BaseModel):
    """Body for POST /agent/{id}/skill."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    requires_approval: bool = False
    security_level_required: int = 1


class SkillUpdateRequest(BaseModel):
    """Body for PUT /agent/{id}/skill/{name}."""
    version: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    parameters: Optional[dict[str, Any]] = None
    requires_approval: Optional[bool] = None
    deprecated: Optional[bool] = None
    security_level_required: Optional[int] = None


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------


@router.get("/registry")
async def get_skill_registry() -> dict[str, Any]:
    """Full registry snapshot."""
    reg = _get_registry()
    return {"registry": reg.get_snapshot()}


@router.get("/search")
async def search_skills(
    query: str = Query(default="", description="Search text"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    agent_id: Optional[str] = Query(default=None, description="Limit to agent"),
) -> dict[str, Any]:
    """Search skills by name, description, or tag."""
    reg = _get_registry()
    results = reg.search(query=query, tag=tag, agent_id=agent_id)
    return {
        "results": [
            {
                "agent_id": r["agent_id"],
                "skill": r["skill"].model_dump(mode="json"),
                "match_score": r["match_score"],
            }
            for r in results
        ],
        "count": len(results),
    }


@router.get("/agent/{agent_id}/skills")
async def get_agent_skills(agent_id: str) -> dict[str, Any]:
    """All skills for a specific agent."""
    reg = _get_registry()
    skills = reg.get_skills(agent_id)
    return {
        "agent_id": agent_id,
        "skills": [s.model_dump(mode="json") for s in skills],
        "count": len(skills),
    }


@router.get("/agent/{agent_id}/skill/{skill_name}")
async def get_skill_detail(agent_id: str, skill_name: str) -> dict[str, Any]:
    """Get details for a single skill."""
    reg = _get_registry()
    skill = reg.get_skill(agent_id, skill_name)
    if skill is None:
        raise HTTPException(
            status_code=404,
            detail=f"Skill {skill_name!r} not found for agent {agent_id!r}",
        )
    return {"agent_id": agent_id, "skill": skill.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/agent/{agent_id}/skill")
async def add_skill(agent_id: str, body: SkillCreateRequest) -> dict[str, Any]:
    """Register a new skill for an agent at runtime."""
    reg = _get_registry()
    agents = _get_agents()

    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")

    deps = [SkillDependency(**d) for d in body.dependencies]
    skill = SkillDefinition(
        name=body.name,
        version=body.version,
        description=body.description,
        parameters=body.parameters,
        tags=body.tags,
        dependencies=deps,
        requires_approval=body.requires_approval,
        security_level_required=body.security_level_required,
        source="api",
        author="api",
    )

    valid, errors = reg.validate(skill)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": errors})

    is_new = reg.register(agent_id, skill)

    # Sync agent identity
    agent = agents[agent_id]
    agent.identity.capabilities = list(reg.get_skills(agent_id))

    return {
        "success": True,
        "created": is_new,
        "skill": skill.model_dump(mode="json"),
    }


@router.put("/agent/{agent_id}/skill/{skill_name}")
async def update_skill(
    agent_id: str, skill_name: str, body: SkillUpdateRequest
) -> dict[str, Any]:
    """Update an existing skill (partial)."""
    reg = _get_registry()
    agents = _get_agents()

    skill = reg.get_skill(agent_id, skill_name)
    if skill is None:
        raise HTTPException(
            status_code=404,
            detail=f"Skill {skill_name!r} not found for agent {agent_id!r}",
        )

    # Apply partial updates
    data = skill.model_dump()
    if body.version is not None:
        data["version"] = body.version
    if body.description is not None:
        data["description"] = body.description
    if body.tags is not None:
        data["tags"] = body.tags
    if body.parameters is not None:
        data["parameters"] = body.parameters
    if body.requires_approval is not None:
        data["requires_approval"] = body.requires_approval
    if body.deprecated is not None:
        data["deprecated"] = body.deprecated
    if body.security_level_required is not None:
        data["security_level_required"] = body.security_level_required

    from datetime import datetime, timezone
    data["updated_at"] = datetime.now(timezone.utc)

    updated = SkillDefinition(**data)
    valid, errors = reg.validate(updated)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": errors})

    reg.register(agent_id, updated)

    # Sync
    if agent_id in agents:
        agents[agent_id].identity.capabilities = list(reg.get_skills(agent_id))

    return {
        "success": True,
        "old_version": skill.version,
        "new_version": updated.version,
        "skill": updated.model_dump(mode="json"),
    }


@router.delete("/agent/{agent_id}/skill/{skill_name}")
async def remove_skill(agent_id: str, skill_name: str) -> dict[str, Any]:
    """Unregister a skill from an agent."""
    reg = _get_registry()
    agents = _get_agents()

    removed = reg.unregister(agent_id, skill_name)
    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"Skill {skill_name!r} not found for agent {agent_id!r}",
        )

    # Sync
    if agent_id in agents:
        agents[agent_id].identity.capabilities = list(reg.get_skills(agent_id))

    return {"success": True, "removed": skill_name}


# ---------------------------------------------------------------------------
# Validation & Dependencies
# ---------------------------------------------------------------------------


@router.post("/validate")
async def validate_skill(body: SkillCreateRequest) -> dict[str, Any]:
    """Validate a skill definition without registering it."""
    reg = _get_registry()
    deps = [SkillDependency(**d) for d in body.dependencies]
    skill = SkillDefinition(
        name=body.name,
        version=body.version,
        description=body.description,
        parameters=body.parameters,
        tags=body.tags,
        dependencies=deps,
        requires_approval=body.requires_approval,
        security_level_required=body.security_level_required,
        source="api",
    )
    valid, errors = reg.validate(skill)
    return {"valid": valid, "errors": errors}


@router.get("/agent/{agent_id}/skill/{skill_name}/deps")
async def get_skill_dependencies(agent_id: str, skill_name: str) -> dict[str, Any]:
    """Recursively resolve all dependencies for a skill."""
    reg = _get_registry()
    try:
        resolved = reg.resolve_dependencies(agent_id, skill_name)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "agent_id": agent_id,
        "skill_name": skill_name,
        "dependencies": {
            name: skill.model_dump(mode="json") for name, skill in resolved.items()
        },
        "count": len(resolved),
    }


# ---------------------------------------------------------------------------
# Task Routing
# ---------------------------------------------------------------------------


@router.post("/tasks/validate-routing")
async def validate_task_routing(
    required_capabilities: list[str],
    version_constraints: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Check if a task with given capabilities can be routed."""
    agents = _get_agents()
    reg = _get_registry()

    available: list[str] = []
    missing: list[str] = []

    for cap_name in required_capabilities:
        found = False
        for aid, agent in agents.items():
            if agent.has_capability(cap_name):
                # Version check
                if version_constraints and cap_name in version_constraints:
                    if reg.agent_has_skill_version(
                        aid, cap_name, version_constraints[cap_name]
                    ):
                        found = True
                        if aid not in available:
                            available.append(aid)
                else:
                    found = True
                    if aid not in available:
                        available.append(aid)
        if not found:
            missing.append(cap_name)

    return {
        "routable": len(missing) == 0,
        "agents_available": available,
        "missing_capabilities": missing,
    }


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


@router.post("/reload")
async def reload_all_skills() -> dict[str, Any]:
    """Re-load skills from YAML for all agents (hot-reload)."""
    agents = _get_agents()
    updated = 0
    errors: list[str] = []

    for agent_id, agent in agents.items():
        try:
            await agent.refresh_skills()
            updated += 1
        except Exception as exc:
            errors.append(f"{agent_id}: {exc}")

    return {"success": len(errors) == 0, "agents_updated": updated, "errors": errors}


@router.post("/agent/{agent_id}/reload")
async def reload_agent_skills(agent_id: str) -> dict[str, Any]:
    """Re-load skills from YAML for a single agent."""
    agents = _get_agents()

    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")

    agent = agents[agent_id]
    await agent.refresh_skills()
    reg = _get_registry()

    return {
        "success": True,
        "agent_id": agent_id,
        "skills_loaded": len(reg.get_skills(agent_id)),
    }


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------


@router.get("/audit-log")
async def get_audit_log(
    agent_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Get audit log of skill changes."""
    reg = _get_registry()
    entries = reg.get_audit_log(agent_id=agent_id, limit=limit)
    return {"entries": entries, "count": len(entries)}
