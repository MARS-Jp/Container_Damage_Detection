"""Guard against misleading conclusions in the training-data audit."""
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tools.audit_shadow_training import audit_dataset, compare_annotations, parse_polygons


class TrainingAuditTests(unittest.TestCase):
    def test_boxes_and_invalid_polygons_are_not_accepted_as_segmentation(self):
        text = '\n'.join(['2 .5 .5 .2 .2', '2 0 0 1 0 nan 1',
                          '2 0 0 1 0 1.2 1', '2 0 0 .5 .5 1 1',
                          '2.5 0 0 1 0 1 1'])
        polygons, issues = parse_polygons(text, {2: 'D'})
        self.assertEqual(polygons, [])
        self.assertEqual([x['issue'] for x in issues], [
            'box_row_not_segmentation', 'non_finite', 'outside_image',
            'degenerate_polygon', 'unknown_class'])

    def test_same_bbox_different_outlines_are_flagged_for_loader_review(self):
        polygons, issues = parse_polygons(
            '2 0 0 1 0 1 1\n2 0 0 0 1 1 1', {2: 'D'})
        self.assertEqual(len(polygons), 2)
        self.assertEqual(issues[0]['issue'], 'duplicate_class_bbox')

    def test_bbox_overlap_is_not_reported_as_mask_coverage(self):
        polygons, _ = parse_polygons('2 0 0 1 0 0 1', {2: 'D'})
        annotation = {'width': 100, 'height': 100, 'objects': [
            {'id': 1, 'class_name': 'D', 'box_xyxy': [80, 80, 90, 90]},
            {'id': 2, 'class_name': 'R', 'box_xyxy': [0, 0, 20, 20]}]}
        rows = compare_annotations(annotation, polygons)
        self.assertGreater(rows[0]['max_same_class_bbox_iou'], 0)
        self.assertEqual(rows[0]['old_mask_pixels_in_xml_box'], 0)
        self.assertEqual(rows[1]['max_same_class_bbox_iou'], 0)
        self.assertEqual(rows[1]['old_mask_pixels_in_xml_box'], 0)

    def test_regions_follow_existing_gt_center_definition(self):
        annotation = {'width': 800, 'height': 600, 'objects': [
            {'id': 84, 'class_name': 'D', 'box_xyxy': [519, 449, 535, 494]},
            {'id': 61, 'class_name': 'D', 'box_xyxy': [60, 180, 100, 220]}]}
        rows = compare_annotations(annotation, [])
        self.assertEqual([x['region'] for x in rows], ['right_shadow', 'left_MSC'])
        self.assertTrue(all(x['review_reason'] == 'no_same_class_bbox_overlap' for x in rows))

    def test_identical_images_with_different_labels_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ('images/train', 'labels/train', 'report'):
                (root / folder).mkdir(parents=True)
            encoded = cv2.imencode('.png', np.zeros((30, 40, 3), np.uint8))[1]
            for name, label in [('a', '2 0 0 1 0 1 1'), ('b', '')]:
                encoded.tofile(root / f'images/train/{name}.png')
                (root / f'labels/train/{name}.txt').write_text(label, encoding='utf-8')
            summary = audit_dataset(root, {2: 'D'}, root / 'report')
            self.assertEqual(len(summary['byte_identical_image_groups']), 1)
            self.assertEqual(len(summary['byte_identical_different_mask_pairs']), 1)
            self.assertEqual(summary['byte_identical_different_mask_pairs'][0]['other_classes'], '')


if __name__ == '__main__':
    unittest.main()
