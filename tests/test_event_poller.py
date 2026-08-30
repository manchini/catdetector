import unittest
from datetime import datetime, timedelta

from tests._stubs import install_requests_stub

install_requests_stub()

from src.dvr.dahua_api import RecordedFile
from src.dvr.event_poller import DVREventPoller


class _FakeClient:
    def __init__(self, events):
        self.events = events
        self.last_call_kwargs: dict = {}

    def list_motion_events(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self.events


class _FakeState:
    def __init__(self, last_seen):
        self.last_seen = last_seen
        self.updates = []

    def get_last_seen(self, dvr_name, channel):
        return self.last_seen

    def update_last_seen(self, dvr_name, channel, ts):
        self.updates.append((dvr_name, channel, ts))


def _recorded_file(start_time):
    return RecordedFile(
        channel=1,
        start_time=start_time,
        end_time=start_time + timedelta(seconds=30),
        file_path="/mnt/dvr/test.dav",
        events=["VideoMotion"],
        length_bytes=1024,
        video_stream="Main",
        dvr_ip="192.0.2.10",
        dvr_name="dvr_test",
    )


class DVREventPollerTests(unittest.TestCase):
    def test_does_not_advance_state_when_callback_returns_false(self):
        last_seen = datetime(2026, 4, 23, 19, 0, 0)
        event = _recorded_file(last_seen + timedelta(minutes=1))
        state = _FakeState(last_seen)
        poller = DVREventPoller(
            client=_FakeClient([event]),
            channels=[1],
            dvr_name="dvr_test",
            on_event=lambda _rec: False,
            state=state,
        )

        poller._poll_once()

        self.assertEqual(state.updates, [])

    def test_advances_state_when_callback_returns_true(self):
        last_seen = datetime(2026, 4, 23, 19, 0, 0)
        event = _recorded_file(last_seen + timedelta(minutes=1))
        state = _FakeState(last_seen)
        poller = DVREventPoller(
            client=_FakeClient([event]),
            channels=[1],
            dvr_name="dvr_test",
            on_event=lambda _rec: True,
            state=state,
        )

        poller._poll_once()

        self.assertEqual(state.updates, [("dvr_test", 1, event.start_time)])

    def test_since_is_passed_to_client(self):
        last_seen = datetime(2026, 4, 23, 19, 0, 0)
        event = _recorded_file(last_seen + timedelta(minutes=1))
        state = _FakeState(last_seen)
        client = _FakeClient([event])
        poller = DVREventPoller(
            client=client,
            channels=[1],
            dvr_name="dvr_test",
            on_event=lambda _rec: True,
            state=state,
        )

        poller._poll_once()

        self.assertIn("since", client.last_call_kwargs)
        self.assertEqual(client.last_call_kwargs["since"], last_seen)

    def test_skips_event_after_max_retries(self):
        last_seen = datetime(2026, 4, 23, 19, 0, 0)
        event = _recorded_file(last_seen + timedelta(minutes=1))
        state = _FakeState(last_seen)
        poller = DVREventPoller(
            client=_FakeClient([event]),
            channels=[1],
            dvr_name="dvr_test",
            on_event=lambda _rec: False,
            state=state,
        )
        poller._max_event_retries = 2

        # Falha 1 e 2 → still blocked
        poller._poll_once()
        poller._poll_once()
        self.assertEqual(state.updates, [])

        # Na 3ª tentativa o evento é pulado e o estado avança
        poller._poll_once()
        self.assertEqual(len(state.updates), 1)
        self.assertEqual(state.updates[0][2], event.start_time)


if __name__ == "__main__":
    unittest.main()
