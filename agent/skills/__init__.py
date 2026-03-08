"""
Skills package — domain-grouped capabilities for the GEM Agent.

Each Skill encapsulates a cohesive set of functions that the Agent can invoke
via prompt-based Function Calling.
"""

from agent.skills.base_skill import BaseSkill, FunctionDefinition, FunctionResult
from agent.skills.skill_registry import SkillRegistry
from agent.skills.knowledge_skill import KnowledgeSkill
from agent.skills.recommend_skill import RecommendSkill
from agent.skills.profile_skill import ProfileSkill
from agent.skills.utility_skill import UtilitySkill

__all__ = [
    "BaseSkill",
    "FunctionDefinition",
    "FunctionResult",
    "SkillRegistry",
    "KnowledgeSkill",
    "RecommendSkill",
    "ProfileSkill",
    "UtilitySkill",
]
