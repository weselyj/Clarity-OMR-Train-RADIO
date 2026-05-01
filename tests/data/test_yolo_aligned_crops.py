"""Unit tests for src.data.yolo_aligned_crops."""
import pytest

from src.data.yolo_aligned_crops import iou_xyxy, match_yolo_to_oracle


class TestIoU:
    def test_identical_boxes_iou_1(self):
        assert iou_xyxy((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_0(self):
        assert iou_xyxy((0, 0, 100, 100), (200, 200, 300, 300)) == 0.0

    def test_half_overlap_iou(self):
        # box B is x:[50,150] y:[0,100]; intersection 50x100=5000; union 15000
        assert iou_xyxy((0, 0, 100, 100), (50, 0, 150, 100)) == pytest.approx(5000 / 15000)

    def test_zero_area_box_iou_0(self):
        assert iou_xyxy((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


class TestMatchYoloToOracle:
    def test_one_to_one_match(self):
        # 2 oracle staves vertically stacked, 2 YOLO bboxes near them
        oracles = [
            {"staff_index": 0, "bbox": (10, 10, 200, 50)},
            {"staff_index": 1, "bbox": (10, 100, 200, 140)},
        ]
        yolo = [
            {"yolo_idx": 0, "bbox": (12, 11, 202, 49), "conf": 0.95},
            {"yolo_idx": 1, "bbox": (10, 101, 199, 140), "conf": 0.90},
        ]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 2
        assert matches[0]["yolo_idx"] == 0 and matches[0]["staff_index"] == 0
        assert matches[1]["yolo_idx"] == 1 and matches[1]["staff_index"] == 1

    def test_yolo_false_positive_dropped(self):
        oracles = [{"staff_index": 0, "bbox": (10, 10, 200, 50)}]
        yolo = [
            {"yolo_idx": 0, "bbox": (10, 10, 200, 50), "conf": 0.95},  # match
            {"yolo_idx": 1, "bbox": (500, 500, 600, 600), "conf": 0.80},  # no overlap
        ]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0]["yolo_idx"] == 0

    def test_oracle_with_no_yolo_match_dropped(self):
        oracles = [
            {"staff_index": 0, "bbox": (10, 10, 200, 50)},
            {"staff_index": 1, "bbox": (10, 100, 200, 140)},  # YOLO misses this
        ]
        yolo = [{"yolo_idx": 0, "bbox": (10, 10, 200, 50), "conf": 0.95}]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0]["staff_index"] == 0
        # Oracle index 1 silently dropped per α-policy

    def test_two_yolo_hit_one_oracle_keep_highest_conf(self):
        oracles = [{"staff_index": 0, "bbox": (10, 10, 200, 50)}]
        yolo = [
            {"yolo_idx": 0, "bbox": (10, 10, 200, 50), "conf": 0.85},
            {"yolo_idx": 1, "bbox": (12, 11, 198, 49), "conf": 0.95},  # higher conf wins
        ]
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 1
        assert matches[0]["yolo_idx"] == 1
        assert matches[0]["staff_index"] == 0

    def test_below_threshold_dropped(self):
        oracles = [{"staff_index": 0, "bbox": (0, 0, 100, 100)}]
        yolo = [{"yolo_idx": 0, "bbox": (50, 50, 150, 150), "conf": 0.95}]  # IoU = 1/7
        matches = match_yolo_to_oracle(yolo, oracles, iou_threshold=0.5)
        assert len(matches) == 0


import textwrap

from src.data.yolo_aligned_crops import load_oracle_bboxes_from_yolo_label


class TestLoadOracleBboxes:
    def test_yolo_label_format_to_xyxy(self, tmp_path):
        # YOLO label format: class cx cy w h, all normalized
        label = tmp_path / "page.txt"
        label.write_text(textwrap.dedent("""\
            0 0.5 0.25 0.4 0.1
            0 0.5 0.75 0.4 0.1
        """))
        oracles = load_oracle_bboxes_from_yolo_label(label, page_width=1000, page_height=1200)
        assert len(oracles) == 2
        # First box: cx=500, cy=300, w=400, h=120 → x1=300, y1=240, x2=700, y2=360
        assert oracles[0]["staff_index"] == 0
        assert oracles[0]["bbox"] == pytest.approx((300.0, 240.0, 700.0, 360.0))
        # Second box: cx=500, cy=900, w=400, h=120
        assert oracles[1]["staff_index"] == 1
        assert oracles[1]["bbox"] == pytest.approx((300.0, 840.0, 700.0, 960.0))

    def test_empty_label_file(self, tmp_path):
        label = tmp_path / "empty.txt"
        label.write_text("")
        assert load_oracle_bboxes_from_yolo_label(label, page_width=100, page_height=100) == []

    def test_staff_indices_assigned_top_to_bottom(self, tmp_path):
        # Provided in non-spatial order; function must sort by y_center to match
        # token_manifest staff_index convention (top staff = staff_index 0).
        label = tmp_path / "page.txt"
        label.write_text(textwrap.dedent("""\
            0 0.5 0.75 0.4 0.1
            0 0.5 0.25 0.4 0.1
        """))
        oracles = load_oracle_bboxes_from_yolo_label(label, page_width=1000, page_height=1200)
        assert oracles[0]["staff_index"] == 0  # the y=0.25 box (top)
        assert oracles[1]["staff_index"] == 1  # the y=0.75 box (bottom)


from PIL import Image

from src.data.yolo_aligned_crops import process_page


class FakeYolo:
    """Stub for ultralytics YOLO that returns a fixed list of bboxes."""
    def __init__(self, predictions):
        self.predictions = predictions  # list of (x1,y1,x2,y2,conf)

    def predict(self, image, **kwargs):
        # Mimic ultralytics interface: returns list with .boxes.xyxy + .boxes.conf
        class Box:
            pass
        class Result:
            pass
        result = Result()
        result.boxes = Box()
        # Provide as plain Python sequences so the matcher can iterate without torch.
        result.boxes.xyxy = [[p[0], p[1], p[2], p[3]] for p in self.predictions]
        result.boxes.conf = [p[4] for p in self.predictions]
        return [result]


class TestProcessPage:
    def test_writes_crops_and_manifest_for_matched_staves(self, tmp_path):
        # 1000×1200 white page with two oracle staves at y=240..360 and y=840..960
        page_img = Image.new("RGB", (1000, 1200), color="white")
        page_path = tmp_path / "page.png"
        page_img.save(page_path)

        label_path = tmp_path / "page.txt"
        label_path.write_text("0 0.5 0.25 0.4 0.1\n0 0.5 0.75 0.4 0.1\n")

        out_crop_dir = tmp_path / "crops"
        out_crop_dir.mkdir()

        # Token sequences — keyed by (page_id, staff_index)
        token_lookup = {
            ("page1", 0): {
                "sample_id": "page1__staff00",
                "page_id": "page1",
                "source_path": "/fake/source.mxl",
                "style_id": "leipzig-default",
                "page_number": 1,
                "staff_index": 0,
                "source_format": "musicxml",
                "score_type": "piano",
                "split": "train",
                "token_sequence": "['<bos>', 'A', '<eos>']",
                "token_count": 3,
            },
            ("page1", 1): {
                "sample_id": "page1__staff01",
                "page_id": "page1",
                "source_path": "/fake/source.mxl",
                "style_id": "leipzig-default",
                "page_number": 1,
                "staff_index": 1,
                "source_format": "musicxml",
                "score_type": "piano",
                "split": "train",
                "token_sequence": "['<bos>', 'B', '<eos>']",
                "token_count": 3,
            },
        }

        # YOLO predicts both staves with offsets that still IoU > 0.5
        fake_yolo = FakeYolo([
            (302, 242, 702, 358, 0.95),
            (300, 842, 700, 958, 0.92),
        ])

        manifest_entries, report = process_page(
            page_id="page1",
            page_image_path=page_path,
            oracle_label_path=label_path,
            yolo_model=fake_yolo,
            token_lookup=token_lookup,
            out_crops_dir=out_crop_dir,
            crop_path_template="data/processed/synthetic_yolo_v1/staff_crops/leipzig-default/{filename}",
            iou_threshold=0.5,
        )

        assert len(manifest_entries) == 2
        assert report["yolo_boxes"] == 2
        assert report["oracle_staves"] == 2
        assert report["matches"] == 2
        assert report["dropped_yolo_fp"] == 0
        assert report["dropped_oracle_recall_gap"] == 0

        # Manifest entries: image_path uses template, staff_index inherited from oracle
        entry0 = next(e for e in manifest_entries if e["staff_index"] == 0)
        assert entry0["sample_id"] == "page1__yoloidx00"
        assert entry0["image_path"].endswith("page1__yoloidx00.png")
        assert entry0["token_sequence"] == "['<bos>', 'A', '<eos>']"
        assert entry0["dataset"] == "synthetic_fullpage"

        # Crops actually written
        assert (out_crop_dir / "page1__yoloidx00.png").exists()
        assert (out_crop_dir / "page1__yoloidx01.png").exists()

    def test_drops_yolo_false_positives_and_recall_gaps(self, tmp_path):
        page_img = Image.new("RGB", (1000, 1200), color="white")
        page_path = tmp_path / "page.png"
        page_img.save(page_path)

        # Two oracle staves
        label_path = tmp_path / "page.txt"
        label_path.write_text("0 0.5 0.25 0.4 0.1\n0 0.5 0.75 0.4 0.1\n")

        token_lookup = {
            ("page1", i): {
                "sample_id": f"page1__staff{i:02d}",
                "page_id": "page1",
                "source_path": "/fake/x.mxl",
                "style_id": "x",
                "page_number": 1,
                "staff_index": i,
                "source_format": "musicxml",
                "score_type": "piano",
                "split": "train",
                "token_sequence": f"['<bos>', '{i}', '<eos>']",
                "token_count": 3,
            } for i in (0, 1)
        }

        # YOLO finds only the first staff + 1 false positive far from anything
        fake_yolo = FakeYolo([
            (302, 242, 702, 358, 0.95),    # match for staff 0
            (10, 1100, 50, 1150, 0.80),    # false positive
        ])

        out_crop_dir = tmp_path / "crops"
        out_crop_dir.mkdir()

        manifest_entries, report = process_page(
            page_id="page1",
            page_image_path=page_path,
            oracle_label_path=label_path,
            yolo_model=fake_yolo,
            token_lookup=token_lookup,
            out_crops_dir=out_crop_dir,
            crop_path_template="x/{filename}",
            iou_threshold=0.5,
        )

        assert len(manifest_entries) == 1
        assert manifest_entries[0]["staff_index"] == 0
        assert report["matches"] == 1
        assert report["dropped_yolo_fp"] == 1
        assert report["dropped_oracle_recall_gap"] == 1
