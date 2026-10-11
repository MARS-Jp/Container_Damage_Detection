# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""YAML-selected training constraints that preserve the segmentation inference graph."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .damage_extent import instance_extent_loss
from .loss import v8MultiLabelSegmentationLoss
from .ops import crop_mask
from .training_aux_ops import consistency_loss, overlap_pair_loss, shadow_view, soft_boundary


def parse_training_aux(value):
    """Validate opt-in YAML settings; reject typos instead of silently dropping an experiment."""
    defaults = {
        "version": 1,
        "shadow": {"enabled": False, "strength": [0.15, 0.45], "probability": 1.0, "softness": 0.15},
        "consistency": {"gain": 0.0, "confidence": 0.7},
        "boundary": {"gain": 0.0},
        "extent": {"gain": 0.0, "fn_weight": 0.7, "margin": 1.0},
        "overlap": {"gain": 0.0, "pairs": [["D", "R"]]},
    }
    if not isinstance(value, dict):
        raise ValueError("training_aux must be a mapping of training-only options.")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown training_aux options: {sorted(unknown)}")
    config = deepcopy(defaults)
    for key, entry in value.items():
        if key == "version":
            if type(entry) is not int or entry != 1:
                raise ValueError("Only training_aux.version: 1 is supported.")
            continue
        if not isinstance(entry, dict) or set(entry) - set(defaults[key]):
            raise ValueError(f"Invalid training_aux.{key} options.")
        config[key].update(deepcopy(entry))

    def number(entry, name, lower=0.0, upper=None, upper_inclusive=True):
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry):
            raise ValueError(f"{name} must be a finite number.")
        if entry < lower or (upper is not None and (entry > upper if upper_inclusive else entry >= upper)):
            raise ValueError(f"{name} is outside its valid range.")

    shadow = config["shadow"]
    if type(shadow["enabled"]) is not bool:
        raise ValueError("training_aux.shadow.enabled must be a YAML boolean.")
    if not isinstance(shadow["strength"], (list, tuple)) or len(shadow["strength"]) != 2:
        raise ValueError("training_aux.shadow.strength requires [minimum, maximum].")
    for strength in shadow["strength"]:
        number(strength, "shadow.strength", upper=1, upper_inclusive=False)
    if shadow["strength"][0] > shadow["strength"][1]:
        raise ValueError("shadow.strength minimum exceeds maximum.")
    number(shadow["probability"], "shadow.probability", upper=1)
    number(shadow["softness"], "shadow.softness", lower=1e-4, upper=1)
    for key in ("consistency", "boundary", "overlap", "extent"):
        number(config[key]["gain"], f"{key}.gain")
    number(config["extent"]["fn_weight"], "extent.fn_weight", upper=1, upper_inclusive=False)
    if config["extent"]["fn_weight"] == 0:
        raise ValueError("extent.fn_weight must be strictly between 0 and 1.")
    number(config["extent"]["margin"], "extent.margin")
    number(config["consistency"]["confidence"], "consistency.confidence", lower=0.5, upper=1, upper_inclusive=False)
    if config["consistency"]["gain"] and not shadow["enabled"]:
        raise ValueError("Consistency requires training_aux.shadow.enabled: true for paired views.")
    pairs = config["overlap"]["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("overlap.pairs requires at least one class pair, e.g. [[D, R]].")
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("Each overlap pair must contain two different classes.")
        if any(type(label) not in (str, int) for label in pair):
            raise ValueError("Overlap classes must be dataset names or integer class IDs.")
    return config


def resolve_overlap_pairs(pairs, names, nc):
    """Resolve dataset class names without hard-coding D=2/R=5 in the implementation."""
    names = names if isinstance(names, dict) else dict(enumerate(names))
    lookup = {str(name): int(index) for index, name in names.items()}
    resolved = []
    for pair in pairs:
        indices = []
        for label in pair:
            index = label if type(label) is int else lookup.get(label)
            if index is None or not 0 <= index < nc:
                raise ValueError(f"Overlap class {label!r} is absent from dataset names {names}.")
            indices.append(index)
        if indices[0] == indices[1]:
            raise ValueError("An overlap pair must resolve to different dataset classes.")
        normalized = tuple(sorted(indices))
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved


class TrainingAuxSegmentationLoss(v8MultiLabelSegmentationLoss):
    """Original multi-label objective plus opt-in instance extent, boundary, and overlap constraints.

    No model references, trainable modules, running teachers, or cached graph tensors are stored here. Extent and
    boundary terms join seg_loss; overlap and paired-view consistency join sem_loss. Validation uses the original
    seven-term objective, without the new training-only regularizers.
    """

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.config = parse_training_aux(model.yaml["training_aux"])
        self.boundary_gain = self.config["boundary"]["gain"]
        self.extent_gain = self.config["extent"]["gain"]
        self.overlap_gain = self.config["overlap"]["gain"]
        self.pairs = (
            resolve_overlap_pairs(self.config["overlap"]["pairs"], model.names, self.nc) if self.overlap_gain else []
        )
        self._training_terms = False
        LOGGER.info(
            f"0918 training-only constraints: shadow={self.config['shadow']['enabled']}, "
            f"consistency={self.config['consistency']['gain']}, boundary={self.boundary_gain}, "
            f"overlap={self.overlap_gain}, extent={self.extent_gain}. Inference graph is unchanged."
        )

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Regularize matched INSTANCE masks, so touching same-class boundaries are representable."""
        standard = super().single_mask_loss(gt_mask, pred, proto, xyxy, area)
        if not self._training_terms or not (self.boundary_gain or self.extent_gain):
            return standard
        with torch.autocast(device_type=proto.device.type, enabled=False):
            logits = torch.einsum("in,nhw->ihw", pred.float(), proto.float())
            target = gt_mask.float()
            if self.extent_gain:
                settings = self.config["extent"]
                standard = standard + self.extent_gain * instance_extent_loss(
                    logits, target, xyxy, settings["fn_weight"], settings["margin"]
                )
            if not self.boundary_gain:
                return standard
            predicted_edges = soft_boundary(logits.sigmoid()[:, None])[:, 0]
            target_edges = soft_boundary(target[:, None])[:, 0]
            # Compute morphology before cropping; include the outside edge of each matched box.
            expanded = xyxy.float() + xyxy.new_tensor([-1, -1, 1, 1]).float()
            predicted_edges = crop_mask(predicted_edges, expanded)
            target_edges = crop_mask(target_edges, expanded)
            valid = target_edges.sum(dim=(1, 2)).gt(0).float()
            intersection = (predicted_edges * target_edges).sum(dim=(1, 2))
            total = predicted_edges.sum(dim=(1, 2)) + target_edges.sum(dim=(1, 2))
            edge_dice = 1 - (2 * intersection + 1e-6) / (total + 1e-6)
            # The band BCE supplies gradients even if initial probabilities are spatially flat.
            band_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * target_edges
            band_bce = band_bce.sum(dim=(1, 2)) / target_edges.sum(dim=(1, 2)).clamp_min(1)
            extra = (0.5 * (edge_dice + band_bce) * valid).sum()
        # The existing positive-anchor normalization and hyp.box segmentation gain are retained.
        return standard + self.boundary_gain * extra

    def loss(self, preds, batch):
        previous = self._training_terms
        self._training_terms = bool(preds.get("_training_aux", False))
        try:
            scaled, detached = super().loss(preds, batch)
            if self._training_terms and self.overlap_gain:
                batch_size = preds["boxes"].shape[0]
                native = self.build_multilabel_target(batch, batch_size, batch["masks"].shape[-2:], torch.float32)
                logits = preds["multilabel_logits"]
                extra = sum(overlap_pair_loss(value, native, self.pairs) for value in logits)
                extra = extra * (self.overlap_gain / len(logits))
                scaled[4] += extra * batch_size
                detached[4] += extra.detach()
            return scaled, detached
        finally:
            self._training_terms = previous


def _slice_predictions(value, start, stop, total):
    """Slice raw segmentation predictions along their batch axis, never matching post-NMS instances."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != total:
            raise ValueError("Paired training requires batch-first raw segmentation prediction tensors.")
        return value[start:stop]
    if isinstance(value, dict):
        return {key: _slice_predictions(item, start, stop, total) for key, item in value.items()}
    if isinstance(value, list):
        return [_slice_predictions(item, start, stop, total) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_predictions(item, start, stop, total) for item in value)
    return value


def training_auxiliary_loss(model, batch, preds=None):
    """Use the standard model/CLI, adding views only inside training loss computation.

    All shadow variants use the same clean+shadow concatenated batch and averaged supervised losses, making Shadow Aug
    versus Consistency differ only by the consistency constraint. One shared forward updates BatchNorm once; no frozen
    or detached model copy is constructed.
    """
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    criterion = model.criterion
    if not isinstance(criterion, TrainingAuxSegmentationLoss):
        raise RuntimeError("training_aux YAML must select TrainingAuxSegmentationLoss.")
    config = criterion.config
    if not model.training:
        outputs = model.predict(batch["img"]) if preds is None else preds
        return criterion(outputs, batch)

    if not config["shadow"]["enabled"]:
        outputs = model.predict(batch["img"]) if preds is None else preds
        raw = {**criterion.parse_output(outputs), "_training_aux": True}
        return criterion(raw, batch)

    if preds is not None:
        raise ValueError("Paired shadow training requires compile=False and model(batch), not precomputed predictions.")
    images = batch["img"]
    shadow = config["shadow"]
    shaded, _ = shadow_view(images, shadow["strength"], shadow["probability"], shadow["softness"])
    batch_size = images.shape[0]
    outputs = criterion.parse_output(model.predict(torch.cat((images, shaded), dim=0)))
    clean = _slice_predictions(outputs, 0, batch_size, 2 * batch_size)
    changed = _slice_predictions(outputs, batch_size, 2 * batch_size, 2 * batch_size)
    clean["_training_aux"] = changed["_training_aux"] = True
    clean_loss, clean_items = criterion(clean, batch)
    changed_loss, changed_items = criterion(changed, batch)
    scaled, detached = 0.5 * (clean_loss + changed_loss), 0.5 * (clean_items + changed_items)
    gain = config["consistency"]["gain"]
    if gain:
        clean_logits, changed_logits = clean["multilabel_logits"], changed["multilabel_logits"]
        extra = images.new_zeros((), dtype=torch.float32)
        for teacher, student in zip(clean_logits, changed_logits):
            target = criterion.build_multilabel_target(batch, batch_size, student.shape[-2:], torch.float32)
            extra = extra + consistency_loss(student, teacher, target, config["consistency"]["confidence"])
        extra = extra * (gain / len(clean_logits))
        scaled[4] += extra * batch_size
        detached[4] += extra.detach()
    return scaled, detached
