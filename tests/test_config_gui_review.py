import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests._stubs import install_pipeline_import_stubs

install_pipeline_import_stubs()

from src.dvr.dahua_api import RecordedFile
from src.tools.config_gui import ConfigGuiState


def workspace_tmp(name: str) -> Path:
    path = Path(".test_tmp") / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_frame() -> "np.ndarray":
    return np.zeros((100, 120, 3), dtype=np.uint8)


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def _make_record(self, channel, start):
        return RecordedFile(
            channel=channel,
            start_time=start,
            end_time=start + timedelta(seconds=10),
            file_path="/mnt/dvr/event.dav",
            events=["VideoMotion"],
            length_bytes=100,
            video_stream="Main",
            dvr_ip="192.0.2.10",
            dvr_name="dvr_test",
        )

    def list_motion_events(self, channel, start, end):
        return [self._make_record(channel, start)]

    def find_media_files(self, channel, start, end, **_kwargs):
        return [self._make_record(channel, start)]

    def download_file(self, file_path, dest):
        Path(dest).write_bytes(b"dav")
        return True


class ConfigGuiReviewTests(unittest.TestCase):
    def test_review_state_lists_and_opens_event_with_mocked_dvr(self):
        tmp = workspace_tmp("config_gui")
        try:
            state = ConfigGuiState(
                tmp / "cameras.yaml",
                config={
                    "dvrs": {
                        "dvr_test": {
                            "host": "192.0.2.10",
                            "http_port": 80,
                            "port": 554,
                            "username": "viewer",
                            "password": "secret",
                        }
                    },
                    "cameras": [
                        {
                            "id": "cam_test",
                            "name": "Test",
                            "dvr": "dvr_test",
                            "channel": 1,
                        }
                    ],
                },
            )
            state.cache_dir = tmp / "cache"
            start = datetime(2026, 4, 23, 20, 0, 0)

            with patch("src.tools.config_gui.DahuaHTTPClient", _FakeClient):
                events = state.list_events("cam_test", start, start + timedelta(minutes=1))

            self.assertEqual(len(events), 1)

            with patch("src.tools.config_gui.DahuaHTTPClient", _FakeClient), patch(
                "src.tools.config_gui.frames_from_video",
                return_value=[(0.0, _fake_frame())],
            ):
                opened = state.open_event(events[0]["id"], frame_interval=1.0)

            self.assertEqual(len(opened["frames"]), 1)
            self.assertTrue(state.frame_path(opened["frames"][0]["id"]).exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
