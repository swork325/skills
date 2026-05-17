"""Skills - A collection of AI-powered tools and utilities.

This package provides a set of skills that can be used by AI agents
to perform various tasks, including code generation, data analysis,
and more.

Personal fork notes:
- Using this for experimenting with custom skill implementations
- See skills/custom/ for personal additions
- Added __author__ update and exposed __license__ in __all__ for easier introspection
- Added __description__ for quick package summary without reading full docstring
- Updated __author__ to reflect personal fork ownership
- Added __version_info__ tuple for easier programmatic version comparisons
- Added __fork_of__ to track upstream origin for reference
- Added __fork_version__ to track my own changes independently of upstream __version__
- Added __contact__ with my GitHub handle for quick reference
"""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))  # e.g. (0, 1, 0)
__fork_version__ = "0.1.0-personal.4"  # my own versioning on top of upstream; bump when I make notable changes
__author__ = "personal fork"  # changed from "Skills Contributors" to reflect this is my fork
__license__ = "Apache-2.0"
__description__ = "A collection of AI-powered tools and utilities for agent workflows."
__fork_of__ = "huggingface/skills"  # upstream repo, useful for tracking drift
__contact__ = "https://github.com/personal"  # my GitHub profile for quick reference

from skills.registry import SkillRegistry
from skills.base import BaseSkill, SkillResult

__all__ = [
    "SkillRegistry",
    "BaseSkill",
    "SkillResult",
    "__version__",
    "__version_info__",
    "__fork_version__",
    "__author__",
    "__license__",
    "__description__",
    "__fork_of__",
    "__contact__",
]
