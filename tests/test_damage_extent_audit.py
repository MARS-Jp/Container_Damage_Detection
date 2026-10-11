"""Read-only annotation comparisons must not merge instances or treat boxes as masks."""

import pytest

from tools.audit_damage_extent import audit


def annotation(tmp_path, boxes):
    objects = "".join(
        f"<object><name>D</name><bndbox><xmin>{x}</xmin><ymin>{y}</ymin>"
        f"<xmax>{r}</xmax><ymax>{b}</ymax></bndbox></object>"
        for x, y, r, b in boxes
    )
    path = tmp_path / "image.xml"
    path.write_text(f"<annotation><size><width>100</width><height>100</height></size>{objects}</annotation>")
    return path


def test_shrunk_polygon_is_unmatched_or_requires_extent_review(tmp_path):
    xml = annotation(tmp_path, [(10, 10, 50, 50)])
    labels = tmp_path / "image.txt"
    labels.write_text("0 .2 .2 .3 .2 .3 .3 .2 .3\n")
    assert len(audit(xml, labels, ["D"])["unmatched_xml"]) == 1
    original = labels.read_bytes()
    result = audit(xml, labels, ["D"], min_iou=0.01)
    assert result["matches"][0]["review_extent"] is True
    assert result["matches"][0]["width_ratio"] == pytest.approx(0.25)
    assert result["matches"][0]["bbox_iou"] == pytest.approx(0.0625)
    assert labels.read_bytes() == original


def test_one_polygon_cannot_match_two_annotations(tmp_path):
    xml = annotation(tmp_path, [(10, 10, 50, 50), (30, 10, 70, 50)])
    labels = tmp_path / "image.txt"
    labels.write_text("0 .1 .1 .7 .1 .7 .5 .1 .5\n")
    result = audit(xml, labels, ["D"])
    assert len(result["matches"]) == 1 and len(result["unmatched_xml"]) == 1


@pytest.mark.parametrize(
    "line",
    [
        "0 .3 .3 .2 .2",  # detection format, not polygon
        "0 nan .1 .5 .1 .5 .5",
        "0 -.1 .1 .5 .1 .5 .5",
        "0.5 .1 .1 .5 .1 .5 .5",
        "2 .1 .1 .5 .1 .5 .5",
    ],
)
def test_bad_polygon_fails(tmp_path, line):
    xml = annotation(tmp_path, [(10, 10, 50, 50)])
    labels = tmp_path / "image.txt"
    labels.write_text(line)
    with pytest.raises(ValueError):
        audit(xml, labels, ["D"])


def test_unknown_class_and_invalid_xml_box_fail(tmp_path):
    labels = tmp_path / "image.txt"
    labels.write_text("")
    with pytest.raises(ValueError, match="class"):
        audit(annotation(tmp_path, [(10, 10, 50, 50)]), labels, ["R"])
    with pytest.raises(ValueError, match="box"):
        audit(annotation(tmp_path, [(50, 10, 10, 50)]), labels, ["D"])
