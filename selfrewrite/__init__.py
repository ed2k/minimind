"""Self-Rewriting: Gradient-free weight diagnosis, editing, and downsizing framework."""

from selfrewrite.config import (
    DOWNSIZE_RUNGS,
    PROFILES,
    ComputeProfile,
    DownsizeRung,
    SessionConfig,
    detect_profile_name,
    get_profile,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ComputeProfile",
    "DownsizeRung",
    "SessionConfig",
    "PROFILES",
    "DOWNSIZE_RUNGS",
    "detect_profile_name",
    "get_profile",
]
