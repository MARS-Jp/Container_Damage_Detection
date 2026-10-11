# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Parameter-free operations for training-only illumination, boundary, and overlap constraints.

These functions do not own modules, parameters, or running state. They operate on existing model
outputs and independent ground-truth masks, and are not needed by the inference graph.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _check_bchw(tensor: torch.Tensor, name: str) -> None:
    """Validate the spatial shape without synchronizing tensor contents from the accelerator."""
    if tensor.ndim != 4 or tensor.shape[1] < 1 or min(tensor.shape[-2:]) < 1:
        raise ValueError(f"{name} must have shape (B, C, H, W) with positive C, H, and W.")


def shadow_view(
    images: torch.Tensor,
    strength: tuple[float, float] = (0.15, 0.45),
    probability: float = 1.0,
    softness: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a soft random half-plane shadow without moving pixels or changing annotation geometry.

    Args:
        images: Floating-point RGB images of shape (B, 3, H, W), normally in [0, 1].
        strength: Minimum and maximum fractional attenuation, each in [0, 1].
        probability: Independent probability of applying a shadow to each image.
        softness: Positive transition width in normalized image coordinates.

    Returns:
        Shadowed images in the input dtype and an FP32 (B, 1, H, W) attenuation mask. The same: multiplier ``1 -
            attenuation`` is applied to every RGB channel. Input tensors are unchanged.
    """
    _check_bchw(images, "images")
    if images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("shadow_view requires floating-point RGB images.")
    if not isinstance(strength, Sequence) or len(strength) != 2:
        raise ValueError("strength must contain a minimum and maximum attenuation.")
    low, high = (float(value) for value in strength)
    probability, softness = float(probability), float(softness)
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1):
        raise ValueError("strength must satisfy 0 <= minimum <= maximum <= 1.")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1.")
    if not math.isfinite(softness) or softness <= 0:
        raise ValueError("softness must be finite and positive.")

    batch_size, _, height, width = images.shape
    with torch.autocast(device_type=images.device.type, enabled=False):
        image_float = images.float()
        if probability == 0 or high == 0:
            attenuation = torch.zeros((batch_size, 1, height, width), device=images.device, dtype=torch.float32)
            return image_float.clamp(0, 1).to(images.dtype), attenuation

        row = torch.linspace(-1, 1, height, device=images.device, dtype=torch.float32).view(1, 1, height, 1)
        column = torch.linspace(-1, 1, width, device=images.device, dtype=torch.float32).view(1, 1, 1, width)
        parameter_shape = (batch_size, 1, 1, 1)
        angle = torch.rand(parameter_shape, device=images.device) * (2 * math.pi)
        offset = torch.rand(parameter_shape, device=images.device) - 0.5
        signed_distance = row * angle.sin() + column * angle.cos() - offset
        soft_region = (signed_distance / softness).sigmoid()
        magnitude = low + (high - low) * torch.rand(parameter_shape, device=images.device)
        selected = (torch.rand(parameter_shape, device=images.device) < probability).float()
        attenuation = soft_region * magnitude * selected
        shadowed = (image_float * (1 - attenuation)).clamp(0, 1)
    return shadowed.to(images.dtype), attenuation


def soft_boundary(probabilities: torch.Tensor) -> torch.Tensor:
    """Return an FP32 differentiable 3x3 dilation-minus-erosion boundary map.

    Replicate padding avoids inventing a boundary around a constant image at the image border. Inputs may be class
    probabilities (B, C, H, W) or independent instance masks (N, 1, H, W).
    """
    _check_bchw(probabilities, "probabilities")
    with torch.autocast(device_type=probabilities.device.type, enabled=False):
        padded = F.pad(probabilities.float(), (1, 1, 1, 1), mode="replicate")
        dilation = F.max_pool2d(padded, kernel_size=3, stride=1)
        erosion = -F.max_pool2d(-padded, kernel_size=3, stride=1)
        return dilation - erosion


def resize_presence(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize an FP32 presence map, preserving positive pixels when reducing spatial resolution.

    Reduction uses adaptive max pooling and expansion uses nearest interpolation. Mixed resizing (one dimension
    shrinking and the other growing) performs reduction before expansion.
    """
    _check_bchw(target, "target")
    if len(size) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in size):
        raise ValueError("size must contain two positive integer dimensions.")
    with torch.autocast(device_type=target.device.type, enabled=False):
        resized = target.float()
        reduced_size = tuple(min(source, destination) for source, destination in zip(target.shape[-2:], size))
        if reduced_size != tuple(target.shape[-2:]):
            resized = F.adaptive_max_pool2d(resized, reduced_size)
        if tuple(resized.shape[-2:]) != tuple(size):
            resized = F.interpolate(resized, size=size, mode="nearest")
        return resized


def _balanced_mean(values: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Average positive and negative regions separately, then average the groups that are present."""
    positive, negative = positive.to(values.dtype), negative.to(values.dtype)
    positive_count, negative_count = positive.sum(), negative.sum()
    positive_mean = (values * positive).sum() / positive_count.clamp_min(1)
    negative_mean = (values * negative).sum() / negative_count.clamp_min(1)
    group_count = positive_count.gt(0).to(values.dtype) + negative_count.gt(0).to(values.dtype)
    return (positive_mean + negative_mean) / group_count.clamp_min(1)


def consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    confidence: float = 0.7,
) -> torch.Tensor:
    """Compare sigmoid outputs only where a detached, confident teacher agrees with ground truth.

    All tensors must have the same (B, C, H, W) shape. Foreground and background are normalized separately so abundant
    background does not dominate. No eligible pixels gives a differentiable zero with respect to the student, and the
    teacher never receives gradients from this loss.
    """
    _check_bchw(student_logits, "student_logits")
    if student_logits.shape != teacher_logits.shape or student_logits.shape != target.shape:
        raise ValueError("student_logits, teacher_logits, and target must have identical BCHW shapes.")
    if student_logits.device != teacher_logits.device or student_logits.device != target.device:
        raise ValueError("student_logits, teacher_logits, and target must be on the same device.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.5 <= confidence <= 1:
        raise ValueError("confidence must be between 0.5 and 1.")
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        student = student_logits.float().sigmoid()
        teacher = teacher_logits.detach().float().sigmoid()
        foreground = target.detach() > 0.5
        positive = foreground & (teacher >= confidence)
        negative = ~foreground & (teacher <= 1 - confidence)
        return _balanced_mean((student - teacher).square(), positive, negative)


def overlap_pair_loss(
    logits: torch.Tensor,
    native_targets: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Supervise the joint probability of class pairs only in images with true native-mask overlap.

    ``native_targets`` contains independent per-class mask unions at the original mask resolution. Intersections and
    unions are formed BEFORE resizing, avoiding false overlaps between nearby classes created by coarse pooling. Each
    pair is evaluated within its GT union and positive and negative regions are balanced. Images without genuine overlap
    contribute exactly zero, so this constraint never requires a dent to have rust. The result averages all configured
    pair losses.
    """
    _check_bchw(logits, "logits")
    _check_bchw(native_targets, "native_targets")
    if logits.shape[:2] != native_targets.shape[:2] or logits.device != native_targets.device:
        raise ValueError("logits and native_targets must share batch size, class count, and device.")
    normalized_pairs = []
    for pair in pairs:
        if len(pair) != 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in pair):
            raise ValueError("Each overlap pair must contain two integer class indices.")
        first, second = pair
        if first == second or min(first, second) < 0 or max(first, second) >= logits.shape[1]:
            raise ValueError("Overlap pairs must contain distinct valid class indices.")
        normalized = tuple(sorted(pair))
        if normalized in normalized_pairs:
            raise ValueError("Overlap pairs must not contain duplicate or reversed-duplicate pairs.")
        normalized_pairs.append(normalized)

    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_float = logits.float()
        if not normalized_pairs:
            return logits_float.sum() * 0
        ground_truth = native_targets.detach() > 0.5
        pair_losses = []
        for first, second in normalized_pairs:
            first_target = ground_truth[:, first : first + 1]
            second_target = ground_truth[:, second : second + 1]
            intersection = first_target & second_target
            union = first_target | second_target
            eligible_image = intersection.flatten(1).any(dim=1).view(-1, 1, 1, 1)
            intersection = resize_presence(intersection, logits.shape[-2:])
            union = resize_presence(union, logits.shape[-2:])
            positive = intersection.gt(0) & eligible_image
            negative = union.gt(0) & intersection.eq(0) & eligible_image

            first_logit = logits_float[:, first : first + 1]
            second_logit = logits_float[:, second : second + 1]
            # logit(sigmoid(a) * sigmoid(b)) = a + b - log(1 + exp(a) + exp(b)).
            normalizer = torch.logsumexp(
                torch.stack((torch.zeros_like(first_logit), first_logit, second_logit), dim=0), dim=0
            )
            joint_logit = first_logit + second_logit - normalizer
            error = F.binary_cross_entropy_with_logits(joint_logit, intersection, reduction="none")
            pair_losses.append(_balanced_mean(error, positive, negative))
        return torch.stack(pair_losses).mean()
