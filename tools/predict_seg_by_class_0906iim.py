"""Compatibility entry point for MMS+ASL checkpoints saved with the legacy ``__main__`` model path.

Use the same arguments as predict_seg_by_class.py. The alias below is needed only while unpickling older 0906-IIM
checkpoints; inference itself uses the standard SegmentationModel implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.nn.tasks import SegmentationModel

# torch.load resolves the class name against the currently executing __main__ module.
sys.modules["__main__"].MMSASLSegmentationModel = SegmentationModel

from tools.predict_seg_by_class import main

if __name__ == "__main__":
    raise SystemExit(main())
