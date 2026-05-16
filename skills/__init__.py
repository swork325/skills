"""Skills - A collection of AI-powered tools and utilities.

This package provides a set of skills that can be used by AI agents
to perform various tasks, including code generation, data analysis,
and more.

Personal fork notes:
- Using this for experimenting with custom skill implementations
- See skills/custom/ for personal additions
"""

__version__ = "0.1.0"
__author__ = "Skills Contributors"
__license__ = "Apache-2.0"

from skills.registry import SkillRegistry
from skills.base import BaseSkill, SkillResult

__all__ = [
    "SkillRegistry",
    "BaseSkill",
    "SkillResult",
    "__version__",
]
