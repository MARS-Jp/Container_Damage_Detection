# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Training-only supervision for the complete visible extent of matched instance masks."""

import math

import torch
import torch.nn.functional as F


def instance_extent_loss(logits, target, boxes, fn_weight=0.7, margin=1.0):
    """Sum balanced BCE/Tversky losses inside expanded GT boxes, in mask-pixel coordinates.

    Args:
        logits (Tensor): Final instance mask logits, shape (N, H, W).
        target (Tensor): Independently annotated instance masks of the same shape, values 0..1.
        boxes (Tensor): GT xyxy boxes in mask-pixel coordinates, shape (N, 4).
        fn_weight (float): Tversky false-negative coefficient; false positives use 1-fn_weight.
        margin (float): ROI expansion in mask pixels, exposing nearby background to the loss.

            Empty foreground contributes differentiable zero. The caller retains the standard mask loss and
            positive-anchor normalization. This function neither dilates labels nor changes predictions at inference; it
            requires reliable full-instance masks, not boxes filled as pseudo masks.
    """
    if logits.ndim != 3 or logits.shape != target.shape or min(logits.shape[-2:]) < 1:
        raise ValueError("Extent logits and targets must have matching (N, H, W) shapes.")
    if boxes.shape != (logits.shape[0], 4):
        raise ValueError("Extent boxes must have shape (N, 4).")
    if target.device != logits.device or boxes.device != logits.device:
        raise ValueError("Extent logits, targets and boxes must share a device.")
    if (
        isinstance(fn_weight, bool)
        or not isinstance(fn_weight, (float, int))
        or not math.isfinite(fn_weight)
        or not 0 < fn_weight < 1
    ):
        raise ValueError("Extent fn_weight must be finite and strictly between 0 and 1.")
    if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("Extent margin must be finite and nonnegative.")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.float()
        y = target.detach().float()
        coords = boxes.detach().float()
        # Pixel-center comparisons are identical on CPU and CUDA and never modify caller tensors.
        x = torch.arange(logits.shape[2], device=logits.device, dtype=torch.float32)[None, None, :] + 0.5
        r = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32)[None, :, None] + 0.5
        x1, y1, x2, y2 = (coords[:, i, None, None] for i in range(4))
        roi = ((x >= x1 - margin) & (x < x2 + margin) & (r >= y1 - margin) & (r < y2 + margin)).float()
        positive, negative = roi * y, roi * (1 - y)
        p = scores.sigmoid()
        tp = (p * positive).sum((1, 2))
        fn = ((1 - p) * positive).sum((1, 2))
        fp = (p * negative).sum((1, 2))
        tversky = 1 - (tp + 1e-6) / (tp + fn_weight * fn + (1 - fn_weight) * fp + 1e-6)
        bce = F.binary_cross_entropy_with_logits(scores, y, reduction="none")
        pos_n, neg_n = positive.sum((1, 2)), negative.sum((1, 2))
        balanced = (bce * positive).sum((1, 2)) / pos_n.clamp_min(1) + (bce * negative).sum((1, 2)) / neg_n.clamp_min(1)
        groups = pos_n.gt(0).float() + neg_n.gt(0).float()
        balanced = balanced / groups.clamp_min(1)
        return (0.5 * (tversky + balanced) * pos_n.gt(0).float()).sum()
