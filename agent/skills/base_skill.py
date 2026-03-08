"""
Base Skill — abstract contract that every Skill must implement.

A *Skill* is a self-contained group of related capabilities (functions) that
the Agent can invoke.  Each Skill:
    • owns one or more *functions* (native callables with JSON-Schema params)
    • manages its own dependencies (retrievers, data files, …)
    • exposes a unified schema so the LLM can discover all functions at once

Design inspired by Semantic Kernel's Skill / Plugin model, adapted for a
prompt-based Function-Calling agent.

Progressive disclosure:
    Layer 1 — ``FunctionResult`` / ``FunctionDefinition``  (data classes)
    Layer 2 — ``BaseSkill``  (abstract, define your own skill)
    Layer 3 — ``SkillRegistry``  (aggregator, used by the Agent)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data classes shared across all skills
# ------------------------------------------------------------------

@dataclass(frozen=True)
class FunctionDefinition:
    """
    Immutable metadata + callable for a single function inside a Skill.

    ``frozen=True`` ensures definitions cannot be accidentally mutated
    after registration.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., str]


@dataclass
class FunctionResult:
    """Uniform wrapper returned after executing any function."""

    function_name: str
    skill_name: str
    success: bool
    data: str
    error: Optional[str] = None


# ------------------------------------------------------------------
# Abstract Skill base class
# ------------------------------------------------------------------

class BaseSkill(ABC):
    """
    Abstract base for all Skills.

    Subclasses must implement:
        • ``name``                — unique skill identifier (property)
        • ``description``         — human-readable summary (property)
        • ``configure``           — inject runtime deps, call ``_finalize()`` at the end
        • ``_register_functions`` — register all functions belonging to this skill

    Usage::

        skill = KnowledgeSkill()
        skill.configure(retriever=..., ...)   # internally calls _finalize()
        schemas = skill.get_function_schemas()
        result  = skill.invoke("search_knowledge_base", {"query": "..."})
    """

    def __init__(self) -> None:
        self._functions: Dict[str, FunctionDefinition] = {}
        self._configured: bool = False

    # ---- Abstract interface -----------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this skill (e.g. 'knowledge')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this skill can do."""
        ...

    @abstractmethod
    def configure(self, **kwargs: Any) -> None:
        """
        Inject runtime dependencies and then call ``self._finalize()``.

        Implementations must end with ``self._finalize()`` to register
        functions and mark the skill as ready.
        """
        ...

    @abstractmethod
    def _register_functions(self) -> None:
        """Register all functions that belong to this skill."""
        ...

    # ---- Lifecycle helpers ------------------------------------------

    def _finalize(self) -> None:
        """
        Standard post-configure hook.

        Call this at the **end** of ``configure()`` to register functions
        and flip the configured flag.  This ensures a consistent lifecycle
        across all skill implementations.
        """
        self._register_functions()
        self._configured = True
        logger.info(
            "Skill '%s' ready (%d functions: %s)",
            self.name, len(self._functions), list(self._functions.keys()),
        )

    # ---- Function registration helpers ------------------------------

    def _add_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., str],
    ) -> None:
        """Register a single function inside this skill."""
        if name in self._functions:
            logger.warning(
                "Skill '%s': function '%s' is being overwritten", self.name, name
            )
        self._functions[name] = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )

    # ---- Schema generation (for LLM prompt injection) ---------------

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON-serialisable schemas for all functions in this skill."""
        return [
            {
                "name": f.name,
                "description": f.description,
                "parameters": f.parameters,
            }
            for f in self._functions.values()
        ]

    def get_function_names(self) -> List[str]:
        """Return the names of all registered functions."""
        return list(self._functions.keys())

    # ---- Invocation -------------------------------------------------

    def invoke(self, function_name: str, arguments: Dict[str, Any]) -> FunctionResult:
        """
        Invoke a function by name, returning a ``FunctionResult``.

        Raises ``RuntimeError`` if the skill has not been configured yet.
        """
        if not self._configured:
            msg = f"Skill '{self.name}' has not been configured — call configure() first"
            logger.error(msg)
            return FunctionResult(
                function_name=function_name,
                skill_name=self.name,
                success=False,
                data="",
                error=msg,
            )

        if function_name not in self._functions:
            msg = f"Skill '{self.name}': unknown function '{function_name}'"
            logger.error(msg)
            return FunctionResult(
                function_name=function_name,
                skill_name=self.name,
                success=False,
                data="",
                error=msg,
            )

        func_def = self._functions[function_name]
        try:
            result_data = func_def.func(**arguments)
            return FunctionResult(
                function_name=function_name,
                skill_name=self.name,
                success=True,
                data=result_data,
            )
        except Exception as exc:
            logger.error(
                "Skill '%s' function '%s' failed: %s",
                self.name, function_name, exc, exc_info=True,
            )
            return FunctionResult(
                function_name=function_name,
                skill_name=self.name,
                success=False,
                data="",
                error=str(exc),
            )

    # ---- Dunder helpers ---------------------------------------------

    def __contains__(self, function_name: str) -> bool:
        return function_name in self._functions

    def __len__(self) -> int:
        return len(self._functions)

    def __repr__(self) -> str:
        status = "ready" if self._configured else "not configured"
        return (
            f"<{self.__class__.__name__} name='{self.name}' "
            f"status={status} functions={self.get_function_names()}>"
        )
