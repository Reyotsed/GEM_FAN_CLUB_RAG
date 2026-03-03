"""
Tool Registry — manages tool definitions and dispatching for the GEM Agent.

Each tool is described by a JSON-serialisable schema that the LLM uses to
decide which tool(s) to call, plus a callable that executes the tool logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Uniform wrapper for a tool execution result."""

    tool_name: str
    success: bool
    data: str
    error: Optional[str] = None


@dataclass
class ToolDefinition:
    """Metadata + callable for a single tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[..., str]


class ToolRegistry:
    """
    Central registry that holds every tool the Agent can use.

    Usage:
        registry = ToolRegistry()
        registry.register(name=..., description=..., parameters=..., func=...)
        result = registry.invoke("search_knowledge_base", {"query": "..."})
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., str],
    ) -> None:
        """Register a new tool."""
        if name in self._tools:
            logger.warning("Tool '%s' is being overwritten", name)
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        logger.debug("Registered tool: %s", name)

    # ------------------------------------------------------------------
    # Schema generation (for LLM prompt injection)
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return a list of tool schemas suitable for embedding in a prompt."""
        schemas: List[Dict[str, Any]] = []
        for t in self._tools.values():
            schemas.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            )
        return schemas

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Invoke a registered tool by name."""
        if tool_name not in self._tools:
            msg = f"Unknown tool: {tool_name}"
            logger.error(msg)
            return ToolResult(tool_name=tool_name, success=False, data="", error=msg)

        tool_def = self._tools[tool_name]
        try:
            result_data = tool_def.func(**arguments)
            return ToolResult(tool_name=tool_name, success=True, data=result_data)
        except Exception as exc:
            logger.error("Tool '%s' execution failed: %s", tool_name, exc, exc_info=True)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data="",
                error=str(exc),
            )

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
