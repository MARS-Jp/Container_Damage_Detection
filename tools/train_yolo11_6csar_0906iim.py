# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Standalone MMS + ASL trainer for yolo11-6csar-0906iim.yaml.

This file intentionally subclasses the existing model and loss instead of modifying shared project sources.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel, yaml_model_load
from ultralytics.utils import RANK
from ultralytics.utils.loss import MultiChannelDiceLoss, v8MultiLabelSegmentationLoss
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_CFG = ROOT / "ultralytics/cfg/models/11_myself/yolo11-6csar-0906iim.yaml"


class AsymmetricLoss(nn.Module):
    """Unreduced ASL for independent per-pixel Bernoulli logits."""

    def __init__(self, gamma_pos: float = 4.0, gamma_neg: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        """Store positive/negative focusing powers and the negative-probability margin."""
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0:
            raise ValueError("ASL gamma values must be non-negative.")
        if not 0 <= clip < 1:
            raise ValueError("ASL clip must be in [0, 1).")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return one ASL value per class and pixel, without reduction."""
        logits, target = logits.float(), target.float()
        positive_probability = logits.sigmoid()
        negative_probability = 1.0 - positive_probability
        if self.clip:
            negative_probability = (negative_probability + self.clip).clamp(max=1.0)

        log_likelihood = target * positive_probability.clamp_min(self.eps).log()
        log_likelihood += (1.0 - target) * negative_probability.clamp_min(self.eps).log()
        probability = target * positive_probability + (1.0 - target) * negative_probability
        gamma = target * self.gamma_pos + (1.0 - target) * self.gamma_neg
        return -(log_likelihood * (1.0 - probability).pow(gamma))


class MMSASLSegmentationLoss(v8MultiLabelSegmentationLoss):
    """Existing multi-label segmentation loss with its MMS BCE term replaced by ASL."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Read all MMS/ASL settings from the new model YAML."""
        super().__init__(model, tal_topk, tal_topk2)
        config = model.yaml.get("mms", {})
        if str(config.get("loss", "asl")).lower() != "asl":
            raise ValueError("yolo11-6csar-0906iim requires mms.loss: asl.")
        self.multilabel_gain = float(config.get("pixel_gain", self.multilabel_gain))
        self.cooccurrence_weight = float(config.get("cooccurrence_weight", self.cooccurrence_weight))
        self.asl = AsymmetricLoss(
            gamma_pos=float(config.get("gamma_pos", 4.0)),
            gamma_neg=float(config.get("gamma_neg", 1.0)),
            clip=float(config.get("clip", 0.05)),
        )
        self.multilabel_dice = MultiChannelDiceLoss(smooth=1)

    def _multi_label_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combine co-occurrence-weighted ASL with the existing class-wise Dice term."""
        asl = self.asl(logits, target)
        overlap = target.sum(dim=1, keepdim=True).gt(1).to(dtype=asl.dtype)
        pixel_weight = 1.0 + self.cooccurrence_weight * overlap
        weighted_asl = (asl * pixel_weight).sum() / (pixel_weight.sum() * self.nc).clamp_min(1.0)
        return 0.5 * weighted_asl + 0.5 * self.multilabel_dice(logits, target)


class MMSASLSegmentationModel(SegmentationModel):
    """Segmentation model whose multi-label MMS branch uses ASL."""

    def __init__(self, cfg=MODEL_CFG, ch: int = 3, nc: int | None = None, verbose: bool = True):
        """Synchronize each independent MMS projection with the dataset class count before parsing the YAML."""
        model_yaml = yaml_model_load(cfg) if isinstance(cfg, (str, Path)) else deepcopy(cfg)
        output_classes = int(nc if nc is not None else model_yaml["nc"])
        model_yaml["nc"] = output_classes
        for layer in model_yaml["backbone"] + model_yaml["head"]:
            if layer[2] == "CBLinear":
                layer[3][0] = [output_classes]
        super().__init__(model_yaml, ch=ch, nc=output_classes, verbose=verbose)

    def init_criterion(self):
        """Select the standalone ASL criterion without changing the shared model class."""
        return MMSASLSegmentationLoss(self)


class MMSASLTrainer(SegmentationTrainer):
    """Segmentation trainer that constructs MMSASLSegmentationModel."""

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Build the new model and optionally load compatible weights."""
        model = MMSASLSegmentationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def save_model(self):
        """Serialize EMA as the standard segmentation class so checkpoints remain portable for inference."""
        ema_model = unwrap_model(self.ema.ema)
        runtime_class = ema_model.__class__
        ema_model.__class__ = SegmentationModel
        try:
            return super().save_model()
        finally:
            ema_model.__class__ = runtime_class


def parse_args() -> argparse.Namespace:
    """Parse the minimal training options; other defaults come from Ultralytics."""
    parser = argparse.ArgumentParser(description="Train yolo11-6csar-0906iim with MMS + ASL.")
    parser.add_argument("--data", required=True, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    """Train with independent instance masks so overlapping class targets remain available."""
    args = parse_args()
    trainer = MMSASLTrainer(
        overrides={
            "model": str(MODEL_CFG),
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "overlap_mask": False,
        }
    )
    trainer.train()


if __name__ == "__main__":
    main()
