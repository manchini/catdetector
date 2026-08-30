import hashlib
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests._stubs import install_requests_stub

install_requests_stub()

from src.dvr.dahua_api import DahuaHTTPClient, _parse_files_response, format_bytes


def workspace_tmp(name: str) -> Path:
    path = Path(".test_tmp") / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class FakeJsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeResponse:
    def __init__(self, status_code=200, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size):
        for chunk in self._chunks:
            yield chunk


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected GET")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DahuaApiTests(unittest.TestCase):
    def test_rpc_login_uses_rpc_credentials(self):
        client = DahuaHTTPClient(
            dvr_ip="192.0.2.10",
            username="viewer",
            password="viewer-pass",
            rpc_username="admin",
            rpc_password="admin-pass",
        )

        posts = []
        session = Mock()

        def fake_post(_url, json, timeout):
            posts.append(json)
            if json["id"] == 1:
                return FakeJsonResponse({
                    "params": {"random": "abc123", "realm": "Login"},
                    "session": "session-id",
                })
            return FakeJsonResponse({"result": True, "session": "session-id"})

        session.post.side_effect = fake_post

        with patch("src.dvr.dahua_api.requests.Session", return_value=session):
            self.assertIs(client._rpc_login(), session)

        expected_pwd_hash = hashlib.md5(
            "admin:Login:admin-pass".encode()
        ).hexdigest().upper()
        expected_final_hash = hashlib.md5(
            f"admin:abc123:{expected_pwd_hash}".encode()
        ).hexdigest().upper()

        self.assertEqual(posts[0]["params"]["userName"], "admin")
        self.assertEqual(posts[1]["params"]["userName"], "admin")
        self.assertEqual(posts[1]["params"]["password"], expected_final_hash)

    def test_parse_files_response_keeps_extra_metadata(self):
        text = "\n".join(
            [
                "items[0].Channel=2",
                "items[0].StartTime=2026-04-23 20:16:06",
                "items[0].EndTime=2026-04-23 20:16:27",
                "items[0].FilePath=/mnt/dvr/20.16.06-20.16.27[M].dav",
                "items[0].Events[0]=VideoMotion",
                "items[0].Flags[0]=Event",
                "items[0].Length=734003",
                "items[0].VideoStream=Main",
                "items[0].Disk=1",
                "items[0].Partition=2",
                "items[0].Cluster=3",
                "items[0].CutLength=21",
            ]
        )

        files = _parse_files_response(text, "192.0.2.10", "dvr_test")

        self.assertEqual(len(files), 1)
        rec = files[0]
        self.assertEqual(rec.channel, 3)
        self.assertEqual(rec.events, ["VideoMotion"])
        self.assertEqual(rec.flags, ["Event"])
        self.assertEqual(rec.length_bytes, 734003)
        self.assertEqual(rec.disk, 1)
        self.assertEqual(rec.partition, 2)
        self.assertEqual(rec.cluster, 3)
        self.assertEqual(rec.cut_length, 21)
        self.assertEqual(rec.raw_fields["FilePath"], "/mnt/dvr/20.16.06-20.16.27[M].dav")

    def test_format_bytes_does_not_round_small_files_to_zero_mb(self):
        self.assertEqual(format_bytes(734003), "716.8 KiB")

    def test_download_file_detailed_success_small_file(self):
        client = DahuaHTTPClient("192.0.2.10", "admin", "pass")
        payload = b"abc123"
        client._rpc_session = FakeSession(
            [
                FakeResponse(
                    status_code=200,
                    headers={"Content-Length": str(len(payload))},
                    chunks=[payload],
                )
            ]
        )

        tmp = workspace_tmp("download_success")
        try:
            dest = tmp / "event.dav"
            outcome = client.download_file_detailed(
                "/event.dav",
                dest,
                expected_size=len(payload),
                max_retries=1,
            )

            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.bytes_written, len(payload))
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(dest.with_name(dest.name + ".part").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_download_file_detailed_retries_after_remote_close(self):
        client = DahuaHTTPClient("192.0.2.10", "admin", "pass")
        payload = b"abcdef"
        session = FakeSession(
            [
                RuntimeError("remote closed"),
                FakeResponse(
                    status_code=200,
                    headers={"Content-Length": str(len(payload))},
                    chunks=[payload],
                ),
            ]
        )
        client._rpc_session = session
        client._get_rpc_session = lambda: session

        tmp = workspace_tmp("download_retry")
        try:
            dest = tmp / "event.dav"
            outcome = client.download_file_detailed(
                "/event.dav",
                dest,
                expected_size=len(payload),
                max_retries=2,
            )

            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.attempts, 2)
            self.assertEqual(dest.read_bytes(), payload)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
