# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Train the 0910 adaptive-IIM, P2, MMS class-query YOLO segmentation model."""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_yolo11_6csar_0906iim import (
    AsymmetricLoss,
    MMSASLSegmentationLoss,
    MMSASLSegmentationModel,
    MMSASLTrainer,
)
from ultralytics.nn import tasks as nn_tasks
from ultralytics.nn.modules.adaptive_iim_0910 import AdaptiveGatedIIMStem0910
from ultralytics.nn.modules.segment_query_0910 import Segment26ClassQuery0910
from ultralytics.utils import RANK

MODEL_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-6csar-0910iim.yaml"


@contextmanager
def registered_0910_modules():
    """Temporarily map existing parser slots to the isolated 0910 modules."""
    original_iim_stem = nn_tasks.IIMStem
    original_multilabel_head = nn_tasks.Segment26MultiLabel
    nn_tasks.IIMStem = AdaptiveGatedIIMStem0910
    nn_tasks.Segment26MultiLabel = Segment26ClassQuery0910
    try:
        yield
    finally:
        nn_tasks.IIMStem = original_iim_stem
        nn_tasks.Segment26MultiLabel = original_multilabel_head


class IIMQueryASLSegmentationLoss(MMSASLSegmentationLoss):
    """Use ASL for MMS pixels and optionally for the main class scores."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        """Read main-class ASL settings without replacing the instance-mask BCE."""
        super().__init__(model, tal_topk, tal_topk2)
        config = model.yaml.get("mms", {})
        self.main_cls_asl_enabled = bool(config.get("main_cls_asl", True))
        if self.main_cls_asl_enabled:
            self.bce = AsymmetricLoss(
                gamma_pos=float(config.get("main_gamma_pos", 1.0)),
                gamma_neg=float(config.get("main_gamma_neg", 4.0)),
                clip=float(config.get("main_clip", 0.05)),
            )


class IIMQueryASLSegmentationModel(MMSASLSegmentationModel):
    """Segmentation model that parses only the new YAML with the 0910 modules."""

    def __init__(self, cfg=MODEL_CFG, ch: int = 3, nc: int | None = None, verbose: bool = True):
        """Build the model without changing the process-wide Ultralytics registry."""
        with registered_0910_modules():
            super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Select ASL for MMS and the configured main classification scope."""
        return IIMQueryASLSegmentationLoss(self)


class IIMQueryASLTrainer(MMSASLTrainer):
    """Construct and save the isolated 0910 model."""

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Build the 0910 architecture and optionally load compatible pretrained tensors."""
        model = IIMQueryASLSegmentationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse reproducible training options while keeping architecture settings in the YAML."""
    parser = argparse.ArgumentParser(description="Train yolo11-6csar-0910iim with adaptive IIM, P2, MMS, and ASL.")
    parser.add_argument("--data", required=True, help="Absolute dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--mask-ratio",
        type=int,
        default=4,
        dest="mask_ratio",
        help="Mask downsample ratio. The P2-aware prototype head keeps standard stride-4 masks.",
    )
    parser.add_argument("--cls-pw", type=float, default=0.5, dest="cls_pw")
    parser.add_argument("--pretrained", type=Path, default=None, help="Optional compatible .pt weights for warm-start.")
    parser.add_argument(
        "--project",
        default=None,
        help="Optional absolute output root. Omit it to use the normal runs/segment directory.",
    )
    parser.add_argument("--name", default="yolo11-6csar-0910iim")
    return parser.parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict:
    """Translate CLI arguments into Ultralytics trainer overrides."""
    if not 0.0 <= args.cls_pw <= 1.0:
        raise ValueError("--cls-pw must be between 0 and 1.")
    if args.mask_ratio != 4:
        raise ValueError("The P2-aware prototype head requires --mask-ratio 4 for the standard validator.")
    overrides = {
        "model": str(MODEL_CFG),
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "patience": args.patience,
        "workers": args.workers,
        "mask_ratio": args.mask_ratio,
        "cls_pw": args.cls_pw,
        "overlap_mask": False,
        "seed": 0,
        "deterministic": True,
        "name": args.name,
    }
    if args.project is not None:
        overrides["project"] = args.project
    if args.pretrained is not None:
        overrides["pretrained"] = str(args.pretrained)
    return overrides


def main() -> None:
    """Train the 0910 model."""
    args = parse_args()
    trainer = IIMQueryASLTrainer(overrides=build_overrides(args))
    trainer.train()


if __name__ == "__main__":
    main()
