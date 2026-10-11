"""Opt-in multi-label postprocessing for the existing segmentation checkpoint.

This does not change the repository's default predictor or model weights. Auxiliary
maps are collected for experiments only; they do not replace instance masks.
"""

from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import nms, ops


class DamageSegmentationPredictor(SegmentationPredictor):
    multi_label = True
    collect_auxiliary = False

    def postprocess(self, preds, img, orig_imgs):
        self.auxiliary_maps = []
        auxiliary = preds[1].get("multilabel_logits") if isinstance(preds[1], dict) else None
        if self.collect_auxiliary and auxiliary:
            # Each head was trained against independent per-class pixel targets.
            # Remove the same letterbox padding used for ordinary retina masks.
            for logits in auxiliary:
                self.auxiliary_maps.append(
                    [
                        ops.scale_masks(logits[index : index + 1].sigmoid(), original.shape[:2])[0].cpu().numpy()
                        for index, original in enumerate(orig_imgs)
                    ]
                )
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        raw = preds[0][0] if isinstance(preds[0], tuple) else preds[0]
        # NMS modifies xywh coordinates in-place; never mutate raw outputs used
        # by another postprocessing experiment.
        detections = nms.non_max_suppression(
            raw.clone(),
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            multi_label=self.multi_label,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(detections, img, orig_imgs, protos)
