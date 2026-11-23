from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from selfrewrite.config import SessionConfig


@dataclass(frozen=True)
class GenerationRecord:
    problem_id: str
    sample_idx: int
    seed: int
    response_text: str
    num_tokens: int


@dataclass
class ReadResult:
    pass_id: int
    arrays: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)


class TargetBackend(ABC):
    """Provides target model weights, seeded forward generation, and hidden-state reads."""

    @property
    @abstractmethod
    def arch_summary(self) -> str:
        """Human-readable architecture summary passed to the editor prompt."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer decoder layers."""

    @abstractmethod
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Map all editable/inspectable target weight names to exact shapes."""

    @abstractmethod
    def export_sandbox_weights_spec(self, dest_dir: Path) -> Path:
        """Write read-only weights spec file (outside WORK/) for sandbox load_original()."""

    @abstractmethod
    def generate_batch(
        self,
        problems: list[Any],
        edit_tensors: dict[str, Any] | None,
        n_samples: int,
        max_tokens: int,
        config: SessionConfig,
        seed_base: int = 0,
    ) -> list[GenerationRecord]:
        """Generate responses with deterministic per-problem seeds under weights theta (+) edit_tensors."""

    @abstractmethod
    def read_activations(
        self,
        texts: list[str],
        layers: list[int] | None,
        target: str,
        pooling: str,
        k: int,
        edit_tensors: dict[str, Any] | None,
        max_tokens: int,
        pass_id: int,
    ) -> ReadResult:
        """Run no-sampling forward pass and return layer hidden states, MLP activations, or log-probs."""


class EditorBackend(ABC):
    """Produces next assistant turn in a self-rewriting conversation (always on frozen editor weights)."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def complete_turn(
        self,
        messages: list[dict[str, str]],
        config: SessionConfig,
        turn_index: int = 0,
    ) -> str:
        """Generate one assistant reply containing reasoning and <=1 fenced action block."""
