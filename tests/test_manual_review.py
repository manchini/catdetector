import sqlite3
import shutil
import unittest
from datetime import datetime
from pathlib import Path

from src.storage import database
from src.storage.manual_review import (
    bbox_px_to_yolo,
    normalize_boxes,
    save_manual_annotation,
)


def workspace_tmp(name: str) -> Path:
    path = Path(".test_tmp") / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ManualReviewStorageTests(unittest.TestCase):
    def test_bbox_px_to_yolo(self):
        result = bbox_px_to_yolo((10, 20, 30, 60), width=100, height=100)
        self.assertEqual(result, (0.2, 0.4, 0.2, 0.4))

    def test_validate_boxes_rejects_empty_annotated_frame(self):
        with self.assertRaises(ValueError):
            normalize_boxes([], width=100, height=100, status="annotated")

    def test_save_manual_annotation_writes_yolo_and_sqlite(self):
        old_db_path = database.DB_PATH
        try:
            root = workspace_tmp("manual_review")
            database.DB_PATH = root / "catdetector.db"
            source = root / "source.jpg"
            source.write_bytes(b"fake-jpeg")

            result = save_manual_annotation(
                source_image_path=source,
                width=100,
                height=100,
                camera_id="cam_test",
                camera_name="Test",
                timestamp=datetime(2026, 4, 23, 20, 0, 0),
                status="annotated",
                boxes=[{"label": "cat", "bbox_px": [10, 20, 30, 60]}],
                source_id="source-frame-1",
                base_dir=root / "manual_review",
            )

            image_path = Path(result["image_path"])
            label_path = Path(result["label_path"])
            classes_path = Path(result["classes_path"])

            self.assertTrue(image_path.exists())
            self.assertEqual(label_path.read_text(encoding="utf-8").strip(), "0 0.200000 0.400000 0.200000 0.400000")
            self.assertEqual(classes_path.read_text(encoding="utf-8").splitlines(), ["cat", "my_dog", "other"])

            conn = sqlite3.connect(database.DB_PATH)
            try:
                row = conn.execute(
                    "SELECT camera_id, user_label, bbox, image_path FROM detections"
                ).fetchone()
            finally:
                conn.close()

            self.assertEqual(row[0], "cam_test")
            self.assertEqual(row[1], "cat")
            self.assertEqual(row[2], "10,20,30,60")
            self.assertEqual(row[3], str(image_path))
        finally:
            database.DB_PATH = old_db_path
            shutil.rmtree(Path(".test_tmp") / "manual_review", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
