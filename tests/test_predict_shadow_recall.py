"""Check crop coverage and annotation-based evaluation without running a model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_shadow_annotations import evaluate, match_boxes, read_annotation
from tools.predict_shadow_recall import tile_boxes


class RecallTests(unittest.TestCase):
    def test_overlapping_tiles_cover_the_entire_context_and_stay_in_bounds(self):
        for bounds in ((5, 7, 97, 106), (0, 0, 2, 2), (440, 150, 688, 588)):
            coverage = np.zeros((bounds[3], bounds[2]), dtype=bool)
            boxes = tile_boxes(bounds, 0.6, 0.25)
            self.assertEqual(len(boxes), len(set(boxes)))
            for x1, y1, x2, y2 in boxes:
                self.assertTrue(bounds[0] <= x1 < x2 <= bounds[2])
                self.assertTrue(bounds[1] <= y1 < y2 <= bounds[3])
                coverage[y1:y2, x1:x2] = True
            self.assertTrue(coverage[bounds[1]:bounds[3], bounds[0]:bounds[2]].all())

    def test_matching_is_class_aware_and_one_to_one(self):
        gt = [{"id": 1, "class_name": "D", "box_xyxy": [10, 10, 30, 30]},
              {"id": 2, "class_name": "R", "box_xyxy": [10, 10, 30, 30]}]
        predictions = [{"class_name": "D", "box_xyxy": [10, 10, 30, 30], "confidence": 0.9},
                       {"class_name": "D", "box_xyxy": [10, 10, 30, 30], "confidence": 0.5}]
        self.assertEqual(match_boxes(gt, predictions), {0: 0})
        rows, missed = evaluate({"objects": gt, "width": 100, "height": 100}, {"detections": predictions})
        full = {row["class"]: row for row in rows if row["region"] == "full"}
        self.assertEqual(full["D"]["precision"], 0.5)
        self.assertEqual(full["D"]["matched"], 1)
        self.assertEqual(full["R"]["missed"], 1)
        self.assertEqual([item["id"] for item in missed], [2])

    def test_empty_predictions_are_all_missed_and_xml_coordinates_are_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.xml"
            path.write_text('<annotation><filename>image.jpg</filename><size><width>100</width><height>80</height></size>'
                            '<object><name>D</name><bndbox><xmin>1</xmin><ymin>2</ymin>'
                            '<xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>', encoding="utf-8")
            annotation = read_annotation(path)
            self.assertEqual(annotation["objects"][0]["box_xyxy"], [0, 1, 20, 30])
            rows, missed = evaluate(annotation, {"detections": []})
            self.assertEqual(rows[0]["recall"], 0)
            self.assertEqual(len(missed), 1)


if __name__ == "__main__":
    unittest.main()
