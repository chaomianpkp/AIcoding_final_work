import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_CORE_DIR = _ROOT / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from mytensor import _C  # noqa: E402

__all__ = ["_C"]

