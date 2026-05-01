"""Unit test for eval/run_stage_a_only.py."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_run_stage_a_writes_jsonl_with_bboxes(tmp_path: Path):
    """Single page, single bbox → manifest JSONL with one record."""
    from eval.run_stage_a_only import run_stage_a

    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"fake pdf bytes")
    out_dir = tmp_path / "out"

    fake_image = MagicMock()
    fake_image.width = 2550
    fake_image.height = 3300

    with patch("eval.run_stage_a_only.YOLO") as MockYOLO, \
         patch("eval.run_stage_a_only._render_pages") as mock_render:
        mock_render.return_value = [fake_image]

        # Mock YOLO predict result: one box at xyxy=[100,200,1800,280], conf=0.9
        mock_box_xyxy = MagicMock()
        mock_box_xyxy.tolist.return_value = [[100.0, 200.0, 1800.0, 280.0]]
        mock_box_conf = MagicMock()
        mock_box_conf.tolist.return_value = [0.9]
        mock_boxes = MagicMock()
        mock_boxes.xyxy = mock_box_xyxy
        mock_boxes.conf = mock_box_conf

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        run_stage_a(
            pdf_path=fake_pdf,
            yolo_weights=Path("fake.pt"),
            out_dir=out_dir,
            dpi=300,
            conf=0.25,
            imgsz=1920,
        )

    manifest = out_dir / "test_stage_a.jsonl"
    assert manifest.exists()
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["piece"] == "test"
    assert rec["page"] == 0
    assert rec["bbox_xyxy"] == [100.0, 200.0, 1800.0, 280.0]
    assert rec["conf"] == 0.9
    assert rec["page_width"] == 2550
    assert rec["page_height"] == 3300


def test_run_stage_a_no_detections_writes_empty_manifest(tmp_path: Path):
    """A page with no YOLO detections should still produce a manifest file (empty),
    so downstream tooling can distinguish 'ran but found nothing' from 'never ran'.
    """
    from eval.run_stage_a_only import run_stage_a

    fake_pdf = tmp_path / "empty.pdf"
    fake_pdf.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    fake_image = MagicMock()
    fake_image.width = 100
    fake_image.height = 100

    with patch("eval.run_stage_a_only.YOLO") as MockYOLO, \
         patch("eval.run_stage_a_only._render_pages") as mock_render:
        mock_render.return_value = [fake_image]

        # Empty results
        empty_boxes = MagicMock()
        empty_boxes.xyxy = MagicMock()
        empty_boxes.xyxy.tolist.return_value = []
        empty_boxes.conf = MagicMock()
        empty_boxes.conf.tolist.return_value = []

        mock_result = MagicMock()
        mock_result.boxes = empty_boxes

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        MockYOLO.return_value = mock_model

        run_stage_a(fake_pdf, Path("fake.pt"), out_dir)

    manifest = out_dir / "empty_stage_a.jsonl"
    assert manifest.exists()
    assert manifest.read_text() == ""


def test_run_stage_a_records_page_index(tmp_path: Path):
    """Multi-page input: each detection records the correct page index."""
    from eval.run_stage_a_only import run_stage_a

    fake_pdf = tmp_path / "multi.pdf"
    fake_pdf.write_bytes(b"fake")
    out_dir = tmp_path / "out"

    fake_p0 = MagicMock(); fake_p0.width = 100; fake_p0.height = 100
    fake_p1 = MagicMock(); fake_p1.width = 100; fake_p1.height = 100

    with patch("eval.run_stage_a_only.YOLO") as MockYOLO, \
         patch("eval.run_stage_a_only._render_pages") as mock_render:
        mock_render.return_value = [fake_p0, fake_p1]

        # Make the model return one bbox per page when called
        def fake_predict(img, **kw):
            mb_xy = MagicMock(); mb_xy.tolist.return_value = [[10.0, 10.0, 90.0, 90.0]]
            mb_conf = MagicMock(); mb_conf.tolist.return_value = [0.5]
            mb = MagicMock(); mb.xyxy = mb_xy; mb.conf = mb_conf
            r = MagicMock(); r.boxes = mb
            return [r]

        mock_model = MagicMock()
        mock_model.predict.side_effect = fake_predict
        MockYOLO.return_value = mock_model

        run_stage_a(fake_pdf, Path("fake.pt"), out_dir)

    manifest = out_dir / "multi_stage_a.jsonl"
    lines = [json.loads(line) for line in manifest.read_text().strip().splitlines()]
    pages_seen = sorted({r["page"] for r in lines})
    assert pages_seen == [0, 1], f"Expected page indices [0,1]; got {pages_seen}"
