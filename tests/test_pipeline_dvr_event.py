import threading
import unittest
from datetime import datetime, timedelta
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests._stubs import install_pipeline_import_stubs

install_pipeline_import_stubs()

from src.dvr.dahua_api import RecordedFile
from src.pipeline import DetectionPipeline


def _recorded_file():
    start_time = datetime(2026, 4, 23, 19, 0, 0)
    return RecordedFile(
        channel=2,
        start_time=start_time,
        end_time=start_time + timedelta(seconds=30),
        file_path="/mnt/dvr/test.dav",
        events=["VideoMotion"],
        length_bytes=1024,
        video_stream="Main",
        dvr_ip="192.0.2.10",
        dvr_name="dvr_test",
    )


class PipelineDVREventTests(unittest.TestCase):
    def test_returns_false_when_download_and_rtsp_fallback_have_no_frames(self):
        pipeline = DetectionPipeline.__new__(DetectionPipeline)
        pipeline.config = {
            "dvrs": {
                "dvr_test": {
                    "host": "192.0.2.10",
                    "port": 554,
                    "username": "viewer",
                    "password": "viewer-pass",
                }
            },
            "cameras": [
                {
                    "id": "cam_test",
                    "name": "Test Camera",
                    "channel": 2,
                    "dvr": "dvr_test",
                    "enabled": True,
                }
            ],
            "detection": {"frame_interval": 1},
        }
        pipeline._dvr_channel_to_cam = {("dvr_test", 2): "cam_test"}
        pipeline._cam_configs = {"cam_test": {"name": "Test Camera", "detection_zone": None}}
        pipeline._dvr_clients = {
            "dvr_test": Mock(
                download_file_detailed=Mock(
                    return_value=SimpleNamespace(
                        ok=False,
                        reason="failed",
                        bytes_written=0,
                        expected_bytes=1024,
                    )
                )
            )
        }
        pipeline.inference_queue = Queue(maxsize=1)
        pipeline._lock = threading.Lock()

        with patch("src.pipeline.extract_frames_from_rtsp_playback", return_value=[]):
            self.assertFalse(pipeline._on_dvr_event(_recorded_file()))


if __name__ == "__main__":
    unittest.main()
