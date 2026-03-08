"""
Skill Registry — central hub that aggregates all Skills and dispatches
function calls to the correct Skill.

This replaces the flat ToolRegistry with a two-level hierarchy::

    SkillRegistry
      ├── KnowledgeSkill   → search_knowledge_base, search_concert_schedule, search_song_info
      ├── RecommendSkill   → get_hot_songs_recommendation
      ├── ProfileSkill     → lookup_artist_profile, lookup_milestones
      └── UtilitySkill     → get_current_datetime

The LLM still sees a flat list of function schemas (for compatibility with
prompt-based Function Calling), but internally each function is owned and
executed by its parent Skill.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.skills.base_skill import BaseSkill, FunctionResult

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Aggregates multiple :class:`BaseSkill` instances and provides a
    unified interface for schema generation and function invocation.

    Usage::

        registry = SkillRegistry()
        registry.add_skill(knowledge_skill)
        registry.add_skill(recommend_skill)

        schemas    = registry.get_all_function_schemas()   # flat list for LLM
        result     = registry.invoke("search_knowledge_base", {"query": "..."})
    """

    def __init__(self) -> None:
        self._skills: Dict[str, BaseSkill] = {}           # skill_name → Skill
        self._function_to_skill: Dict[str, str] = {}      # function_name → skill_name

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def add_skill(self, skill: BaseSkill) -> None:
        """
        Register a Skill and index all its functions.

        The skill **must** have been configured (``configure()`` called)
        before it can be added.  An unconfigured skill is rejected with a
        ``ValueError``.
        """
        if not skill._configured:
            raise ValueError(
                f"Cannot add unconfigured skill '{skill.name}'. "
                f"Call skill.configure(...) before adding it to the registry."
            )

        if skill.name in self._skills:
            # Clean up stale function→skill mappings from the old version
            old_skill = self._skills[skill.name]
            for old_fn in old_skill.get_function_names():
                self._function_to_skill.pop(old_fn, None)
            logger.warning("Skill '%s' is being replaced", skill.name)

        self._skills[skill.name] = skill

        for fn_name in skill.get_function_names():
            if fn_name in self._function_to_skill:
                old_owner = self._function_to_skill[fn_name]
                logger.warning(
                    "Function '%s' was in skill '%s', now remapped to '%s'",
                    fn_name, old_owner, skill.name,
                )
            self._function_to_skill[fn_name] = skill.name

        logger.info(
            "Registered skill '%s' with functions: %s",
            skill.name, skill.get_function_names(),
        )

    def get_skill(self, skill_name: str) -> Optional[BaseSkill]:
        """Look up a skill by name."""
        return self._skills.get(skill_name)

    def get_skill_names(self) -> List[str]:
        """Return names of all registered skills."""
        return list(self._skills.keys())

    # ------------------------------------------------------------------
    # Schema generation (flat list — compatible with existing LLM prompt)
    # ------------------------------------------------------------------

    def get_all_function_schemas(self) -> List[Dict[str, Any]]:
        """
        Return a flat list of function schemas across all skills.

        This is what gets injected into the LLM planning prompt, so the
        agent sees every function regardless of which skill owns it.
        """
        schemas: List[Dict[str, Any]] = []
        for skill in self._skills.values():
            schemas.extend(skill.get_function_schemas())
        return schemas

    def get_all_function_names(self) -> List[str]:
        """Return names of all functions across all skills."""
        return list(self._function_to_skill.keys())

    # ------------------------------------------------------------------
    # Invocation (dispatches to the owning Skill)
    # ------------------------------------------------------------------

    def invoke(self, function_name: str, arguments: Dict[str, Any]) -> FunctionResult:
        """
        Invoke a function by name.

        The registry looks up which Skill owns the function and delegates
        execution to that Skill.
        """
        skill_name = self._function_to_skill.get(function_name)
        if skill_name is None:
            msg = f"Unknown function: '{function_name}'"
            logger.error(msg)
            return FunctionResult(
                function_name=function_name,
                skill_name="unknown",
                success=False,
                data="",
                error=msg,
            )

        skill = self._skills[skill_name]
        return skill.invoke(function_name, arguments)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, function_name: str) -> bool:
        return function_name in self._function_to_skill

    def __len__(self) -> int:
        """Total number of functions across all skills."""
        return len(self._function_to_skill)

    def __repr__(self) -> str:
        parts = [
            f"  {s.name}: {s.get_function_names()}"
            for s in self._skills.values()
        ]
        return "SkillRegistry(\n" + "\n".join(parts) + "\n)"
