"""Base classes and interfaces for skills.

This module defines the foundational abstractions used across all skills,
including the base Skill class, input/output schemas, and skill metadata.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar


T = TypeVar("T", bound="BaseSkill")


@dataclass
class SkillMetadata:
    """Metadata describing a skill's capabilities and requirements."""

    name: str
    description: str
    version: str = "0.1.0"
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    requires_gpu: bool = False
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "author": self.author,
            "requires_gpu": self.requires_gpu,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


@dataclass
class SkillResult:
    """Encapsulates the output of a skill execution."""

    output: Any
    skill_name: str
    elapsed_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True if the skill executed without errors."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a dictionary."""
        return {
            "output": self.output,
            "skill_name": self.skill_name,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
            "error": self.error,
            "success": self.success,
        }


class BaseSkill(abc.ABC):
    """Abstract base class that all skills must implement.

    Subclasses should override `metadata` and implement `_run`.

    Example::

        class MySkill(BaseSkill):
            metadata = SkillMetadata(
                name="my_skill",
                description="Does something useful.",
            )

            def _run(self, inputs: Dict[str, Any]) -> Any:
                return inputs["text"].upper()
    """

    #: Override this in subclasses to describe the skill.
    metadata: SkillMetadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure concrete subclasses declare metadata.
        if not abc.ABC in cls.__bases__ and not hasattr(cls, "metadata"):
            raise TypeError(
                f"Skill '{cls.__name__}' must define a class-level `metadata` attribute."
            )

    @abc.abstractmethod
    def _run(self, inputs: Dict[str, Any]) -> Any:
        """Execute the skill logic.

        Args:
            inputs: A dictionary of input values validated against
                    ``metadata.input_schema`` (if provided).

        Returns:
            The raw output of the skill.
        """

    def run(self, inputs: Optional[Dict[str, Any]] = None) -> SkillResult:
        """Public entry point that wraps ``_run`` with timing and error handling.

        Args:
            inputs: Input dictionary forwarded to ``_run``.

        Returns:
            A :class:`SkillResult` containing the output or error information.
        """
        inputs = inputs or {}
        start = time.perf_counter()
        try:
            output = self._run(inputs)
            error = None
        except Exception as exc:  # noqa: BLE001
            output = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        return SkillResult(
            output=output,
            skill_name=self.metadata.name,
            elapsed_seconds=round(elapsed, 4),
            error=error,
        )

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        """Instantiate a skill from a configuration dictionary.

        Override this in subclasses that require constructor arguments.
        """
        return cls()  # type: ignore[call-arg]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.metadata.name!r} version={self.metadata.version!r}>"
