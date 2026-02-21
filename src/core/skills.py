"""Skill management system for Code Horde agents.

Provides a centralized registry for agent skills with YAML-driven loading,
semantic versioning, dependency resolution, and runtime management.
"""

from __future__ import annotations

import pathlib
import re
import threading
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import structlog
import yaml
from pydantic import BaseModel, Field

from src.core.agent_base import AgentCapability

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SkillDependency(BaseModel):
    """Dependency on another skill or bridge.

    Attributes:
        skill_name: Name of the required skill.
        version_range: Semver range constraint (e.g. ``>=1.0.0,<2.0.0``).
        optional: If ``True`` the skill works without this dependency.
        bridge_required: Name of a bridge that must be available (e.g. ``github``).
    """

    skill_name: str = Field(description="Required skill name")
    version_range: str = Field(default="*", description="Semver range (e.g. >=1.0.0,<2.0.0)")
    optional: bool = Field(default=False, description="Whether this dependency is optional")
    bridge_required: Optional[str] = Field(default=None, description="Required bridge name")


class SkillDefinition(AgentCapability):
    """Extended capability definition with metadata, tags, and dependencies.

    Inherits from :class:`AgentCapability` so it is a drop-in replacement
    everywhere the base model is accepted.

    Attributes:
        tags: Free-form labels for search/filtering.
        dependencies: Skills or bridges this skill depends on.
        requires_approval: Whether human approval is needed before execution.
        deprecated: Mark the skill as deprecated.
        security_level_required: Minimum agent security level (1-5).
        author: Who created / last updated this skill.
        source: Where the skill definition originated.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last modification.
    """

    tags: list[str] = Field(default_factory=list, description="Categorisation tags")
    dependencies: list[SkillDependency] = Field(default_factory=list, description="Skill dependencies")
    requires_approval: bool = Field(default=False, description="Needs human approval")
    deprecated: bool = Field(default=False, description="Deprecated flag")
    security_level_required: int = Field(default=1, ge=1, le=5, description="Minimum security level")
    author: str = Field(default="system", description="Author or source")
    source: Literal["yaml", "hardcoded", "api"] = Field(default="hardcoded", description="Definition origin")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Semantic Version Helpers
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-[\w.]+)?(?:\+[\w.]+)?$")


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semver string into a ``(major, minor, patch)`` tuple.

    Raises:
        ValueError: If *version* is not valid semver.
    """
    m = _SEMVER_RE.match(version.strip())
    if not m:
        raise ValueError(f"Invalid semver: {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compare_versions(v1: str, v2: str) -> int:
    """Compare two semver strings.

    Returns:
        -1 if *v1* < *v2*, 0 if equal, 1 if *v1* > *v2*.
    """
    a, b = parse_semver(v1), parse_semver(v2)
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


_RANGE_TOKEN_RE = re.compile(r"([><=!]+)\s*([\d]+\.[\d]+\.[\d]+[\w.+-]*)")


def version_satisfies(version: str, range_spec: str) -> bool:
    """Check whether *version* satisfies *range_spec*.

    Supports ``*`` (anything), a plain version (exact match), and
    comma-separated constraints like ``>=1.0.0,<2.0.0``.
    """
    range_spec = range_spec.strip()
    if range_spec in ("*", ""):
        return True

    ver = parse_semver(version)

    # Plain version → exact match
    if _SEMVER_RE.match(range_spec):
        return ver == parse_semver(range_spec)

    for token in range_spec.split(","):
        token = token.strip()
        m = _RANGE_TOKEN_RE.match(token)
        if not m:
            continue
        op, target_str = m.group(1), m.group(2)
        target = parse_semver(target_str)

        if op == ">=" and not (ver >= target):
            return False
        elif op == ">" and not (ver > target):
            return False
        elif op == "<=" and not (ver <= target):
            return False
        elif op == "<" and not (ver < target):
            return False
        elif op == "==" and not (ver == target):
            return False
        elif op == "!=" and not (ver != target):
            return False

    return True


# ---------------------------------------------------------------------------
# Audit Entry
# ---------------------------------------------------------------------------


class _AuditEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: str
    agent_id: str
    skill_name: str
    detail: str = ""


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Thread-safe, in-memory registry of all agent skills.

    The registry is keyed by ``(agent_id, skill_name)`` and supports YAML
    loading, runtime registration, search, dependency resolution, and auditing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # agent_id -> {skill_name -> SkillDefinition}
        self._skills: dict[str, dict[str, SkillDefinition]] = {}
        self._audit_log: list[_AuditEntry] = []

    # -- Registration -------------------------------------------------------

    def register(self, agent_id: str, skill: SkillDefinition) -> bool:
        """Register or update a skill for *agent_id*.

        Returns ``True`` if the skill was newly registered, ``False`` if it
        was updated.
        """
        with self._lock:
            bucket = self._skills.setdefault(agent_id, {})
            is_new = skill.name not in bucket
            bucket[skill.name] = skill
            self._audit_log.append(
                _AuditEntry(
                    action="register" if is_new else "update",
                    agent_id=agent_id,
                    skill_name=skill.name,
                    detail=f"v{skill.version} source={skill.source}",
                )
            )
            return is_new

    def unregister(self, agent_id: str, skill_name: str) -> bool:
        """Remove a skill from *agent_id*.  Returns ``True`` if it existed."""
        with self._lock:
            bucket = self._skills.get(agent_id, {})
            if skill_name in bucket:
                del bucket[skill_name]
                self._audit_log.append(
                    _AuditEntry(action="unregister", agent_id=agent_id, skill_name=skill_name)
                )
                return True
            return False

    def clear_agent(self, agent_id: str) -> int:
        """Remove **all** skills for *agent_id*.  Returns count removed."""
        with self._lock:
            bucket = self._skills.pop(agent_id, {})
            count = len(bucket)
            if count:
                self._audit_log.append(
                    _AuditEntry(
                        action="clear",
                        agent_id=agent_id,
                        skill_name="*",
                        detail=f"removed {count} skills",
                    )
                )
            return count

    # -- Queries ------------------------------------------------------------

    def get_skill(self, agent_id: str, skill_name: str) -> Optional[SkillDefinition]:
        """Look up a single skill."""
        with self._lock:
            return self._skills.get(agent_id, {}).get(skill_name)

    def get_skills(self, agent_id: str) -> list[SkillDefinition]:
        """Return all skills for *agent_id*."""
        with self._lock:
            return list(self._skills.get(agent_id, {}).values())

    def get_all_agents(self) -> list[str]:
        """Return agent IDs that have registered skills."""
        with self._lock:
            return list(self._skills.keys())

    def search(
        self,
        query: str = "",
        tag: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search skills by name/description substring and optional tag.

        Returns a list of ``{agent_id, skill, match_score}`` dicts, sorted
        by match score descending.
        """
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        with self._lock:
            agents = [agent_id] if agent_id else list(self._skills.keys())
            for aid in agents:
                for skill in self._skills.get(aid, {}).values():
                    # Tag filter
                    if tag and tag not in skill.tags:
                        continue

                    # Text matching score
                    score = 0.0
                    if query_lower:
                        if query_lower == skill.name.lower():
                            score = 1.0
                        elif query_lower in skill.name.lower():
                            score = 0.8
                        elif query_lower in skill.description.lower():
                            score = 0.5
                        elif any(query_lower in t.lower() for t in skill.tags):
                            score = 0.4
                        else:
                            continue  # No match at all
                    else:
                        score = 0.5  # No query → return everything

                    results.append({"agent_id": aid, "skill": skill, "match_score": score})

        results.sort(key=lambda r: r["match_score"], reverse=True)
        return results

    # -- Dependency Resolution ----------------------------------------------

    def resolve_dependencies(
        self, agent_id: str, skill_name: str, _visited: Optional[set[str]] = None
    ) -> dict[str, SkillDefinition]:
        """Recursively resolve all dependencies for a skill.

        Returns a ``{skill_name: SkillDefinition}`` mapping of **all** direct
        and transitive dependencies.  Only considers the *same* agent.

        Raises:
            KeyError: If *skill_name* is not registered for *agent_id*.
            ValueError: On circular dependencies.
        """
        if _visited is None:
            _visited = set()

        if skill_name in _visited:
            raise ValueError(f"Circular dependency detected: {skill_name}")
        _visited.add(skill_name)

        skill = self.get_skill(agent_id, skill_name)
        if skill is None:
            raise KeyError(f"Skill {skill_name!r} not found for agent {agent_id!r}")

        resolved: dict[str, SkillDefinition] = {}
        for dep in skill.dependencies:
            dep_skill = self.get_skill(agent_id, dep.skill_name)
            if dep_skill is None:
                if dep.optional:
                    continue
                raise KeyError(
                    f"Required dependency {dep.skill_name!r} (for {skill_name!r}) "
                    f"not found on agent {agent_id!r}"
                )

            # Version check
            if not version_satisfies(dep_skill.version, dep.version_range):
                raise ValueError(
                    f"Dependency {dep.skill_name} v{dep_skill.version} does not "
                    f"satisfy {dep.version_range} (required by {skill_name})"
                )

            resolved[dep.skill_name] = dep_skill
            # Recurse
            transitive = self.resolve_dependencies(agent_id, dep.skill_name, _visited)
            resolved.update(transitive)

        return resolved

    # -- Validation ---------------------------------------------------------

    def validate(self, skill: SkillDefinition) -> tuple[bool, list[str]]:
        """Validate a :class:`SkillDefinition` structurally.

        Returns ``(valid, errors)`` where *errors* is a list of human-readable
        messages.
        """
        errors: list[str] = []

        if not skill.name or not skill.name.strip():
            errors.append("Skill name is required")

        try:
            parse_semver(skill.version)
        except ValueError:
            errors.append(f"Invalid version: {skill.version!r}")

        for dep in skill.dependencies:
            if dep.version_range not in ("*", ""):
                try:
                    # Quick sanity: parse each constraint target
                    for tok in dep.version_range.split(","):
                        tok = tok.strip()
                        if _SEMVER_RE.match(tok):
                            parse_semver(tok)
                except ValueError:
                    errors.append(f"Bad version range in dependency {dep.skill_name!r}: {dep.version_range!r}")

        if skill.security_level_required < 1 or skill.security_level_required > 5:
            errors.append("security_level_required must be 1-5")

        return (len(errors) == 0, errors)

    # -- YAML Loading -------------------------------------------------------

    def load_agent_yaml(self, agent_id: str, yaml_path: pathlib.Path) -> tuple[int, list[str]]:
        """Load skills from an agent's YAML config file.

        The YAML is expected to have a ``skills:`` mapping at the top level.
        Each key under ``skills`` is the skill name.

        Returns ``(loaded_count, errors)``
        """
        errors: list[str] = []

        if not yaml_path.exists():
            return (0, [f"File not found: {yaml_path}"])

        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return (0, [f"YAML parse error: {exc}"])

        if not isinstance(raw, dict):
            return (0, ["YAML root is not a mapping"])

        skills_section = raw.get("skills", {})
        if not isinstance(skills_section, dict):
            return (0, ["'skills' section is not a mapping"])

        loaded = 0
        for skill_name, skill_data in skills_section.items():
            if not isinstance(skill_data, dict):
                errors.append(f"Skill {skill_name!r}: value is not a mapping")
                continue

            try:
                deps: list[SkillDependency] = []
                for d in skill_data.get("dependencies", []):
                    if isinstance(d, dict):
                        deps.append(SkillDependency(**d))

                params: dict[str, Any] = skill_data.get("parameters", {})

                skill = SkillDefinition(
                    name=skill_name,
                    version=str(skill_data.get("version", "1.0.0")),
                    description=str(skill_data.get("description", "")),
                    parameters=params,
                    tags=skill_data.get("tags", []),
                    dependencies=deps,
                    requires_approval=bool(skill_data.get("requires_approval", False)),
                    deprecated=bool(skill_data.get("deprecated", False)),
                    security_level_required=int(skill_data.get("security_level_required", 1)),
                    author=str(skill_data.get("author", "yaml")),
                    source="yaml",
                )

                valid, verrors = self.validate(skill)
                if not valid:
                    errors.extend(f"Skill {skill_name!r}: {e}" for e in verrors)
                    continue

                self.register(agent_id, skill)
                loaded += 1

            except Exception as exc:
                errors.append(f"Skill {skill_name!r}: {exc}")

        return (loaded, errors)

    # -- Snapshot / Serialisation -------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        """Return the full registry state as a JSON-serialisable dict."""
        with self._lock:
            snapshot: dict[str, Any] = {}
            for agent_id, skills in self._skills.items():
                snapshot[agent_id] = {
                    name: skill.model_dump(mode="json") for name, skill in skills.items()
                }
            return snapshot

    def get_audit_log(
        self, agent_id: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return recent audit entries."""
        with self._lock:
            entries = self._audit_log
            if agent_id:
                entries = [e for e in entries if e.agent_id == agent_id]
            return [e.model_dump(mode="json") for e in entries[-limit:]]

    # -- Helpers for Orchestrator -------------------------------------------

    def agent_has_skill(self, agent_id: str, skill_name: str) -> bool:
        """Fast lookup: does *agent_id* have *skill_name*?"""
        with self._lock:
            return skill_name in self._skills.get(agent_id, {})

    def agent_has_skill_version(
        self, agent_id: str, skill_name: str, version_range: str
    ) -> bool:
        """Check that *agent_id* has *skill_name* matching *version_range*."""
        with self._lock:
            skill = self._skills.get(agent_id, {}).get(skill_name)
            if skill is None:
                return False
            return version_satisfies(skill.version, version_range)

    def __repr__(self) -> str:
        with self._lock:
            total = sum(len(s) for s in self._skills.values())
        return f"<SkillRegistry agents={len(self._skills)} skills={total}>"
