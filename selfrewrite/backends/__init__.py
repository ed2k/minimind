"""Target and Editor backends for self-rewriting sessions."""

from selfrewrite.backends.base import EditorBackend, GenerationRecord, ReadResult, TargetBackend
from selfrewrite.backends.mock_backend import (
    MockFailureEditor,
    MockTargetBackend,
    MockTransformerModel,
)

__all__ = [
    "GenerationRecord",
    "ReadResult",
    "TargetBackend",
    "EditorBackend",
    "MockTransformerModel",
    "MockTargetBackend",
    "MockFailureEditor",
]
