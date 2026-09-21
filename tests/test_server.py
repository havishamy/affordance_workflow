import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import requests

from affordance.camera import CapturedFrame
from affordance.config import ServerConfig
from affordance.server import LocalizationClient, build_upload_files


def sample_frame():
    return CapturedFrame(
        np.zeros((2, 3, 3), dtype=np.uint8), np.full((2, 3), 1000, dtype=np.uint16),
        {"width": 3, "height": 2, "fx": 100, "fy": 100, "cx": 1, "cy": 1},
        0.001, "test-camera", 123.0,
    )


def world_result():
    return {
        "job_id": "test-job", "frame": "world", "units": "meters",
        "dropper_bulb_center_world": [0.3, 0.1, 0.2],
        "beaker_mouth_center_world": [0.6, 0.1, 0.2],
    }


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    result._content_consumed = True
    return result


def bundle_response():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("04_targets/titration_targets.json", json.dumps(world_result()))
    result = requests.Response()
    result.status_code = 200
    result.raw = io.BytesIO(buffer.getvalue())
    return result


class ServerResponseTests(unittest.TestCase):
    def test_nested_response_and_mm_conversion(self):
        parsed = LocalizationClient._parse_response(
            {
                "status": "ok",
                "result": {
                    "coordinate_frame": "camera",
                    "unit": "mm",
                    "rubber_bulb": {"position": [10, 20, 300], "confidence": 0.9},
                    "beaker_mouth": {"position": [100, 20, 400], "confidence": 0.8},
                },
            }
        )
        np.testing.assert_allclose(parsed.rubber_bulb.position, [0.01, 0.02, 0.3])
        self.assertEqual(parsed.beaker_mouth.frame, "camera")

    def test_rejects_missing_point(self):
        with self.assertRaises(ValueError):
            LocalizationClient._parse_response({"rubber_bulb": [0, 0, 1]})

    def test_world_is_base_without_fabricated_confidence(self):
        parsed = LocalizationClient._parse_world_response(world_result())
        np.testing.assert_allclose(parsed.rubber_bulb.position, [0.3, 0.1, 0.2])
        self.assertEqual(parsed.rubber_bulb.frame, "base")
        self.assertIsNone(parsed.rubber_bulb.confidence)

    def test_world_rejects_wrong_frame_units_or_points(self):
        for bad in (
            {"frame": "camera"}, {"units": "mm"}, {"units": None},
            {"dropper_bulb_center_world": [1, 2]},
            {"beaker_mouth_center_world": [float("nan"), 0, 0]},
            {"beaker_mouth_center_world": None},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                LocalizationClient._parse_world_response({**world_result(), **bad})


class JobClientTests(unittest.TestCase):
    def setUp(self):
        self.config = ServerConfig("http://localhost:8000/", 2, 1, True, "TEST_LAG_TOKEN", 0.001, 10)
        self.files = build_upload_files(sample_frame(), np.eye(4))
        self.env = patch.dict("os.environ", {"TEST_LAG_TOKEN": "test-only-token"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_submit_poll_result_and_bundle(self):
        with tempfile.TemporaryDirectory() as root, patch("affordance.server.requests.post") as post, patch("affordance.server.requests.get") as get:
            post.return_value = response({"job_id": "test-job", "state": "queued"})
            get.side_effect = [
                response({"job_id": "test-job", "state": "queued"}),
                response({"job_id": "test-job", "state": "running"}),
                response({"job_id": "test-job", "state": "completed"}),
                response(world_result()), bundle_response(),
            ]
            parsed = LocalizationClient(self.config).locate_files(self.files, 0.001, "capture-test", Path(root))
            self.assertEqual(parsed.beaker_mouth.frame, "base")
            post.assert_called_once()
            args, kwargs = post.call_args
            self.assertEqual(args[0], "http://localhost:8000/v1/titration/jobs")
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-only-token")
            self.assertEqual(kwargs["data"], {"depth_scale": "0.001", "capture_id": "capture-test"})
            # Exercise requests' real multipart encoder, including PNG bytes and JSON file parts.
            prepared = requests.Request("POST", args[0], files=kwargs["files"], data=kwargs["data"]).prepare()
            for field in ("rgb", "depth", "camera", "world_ee"):
                self.assertIn(f'name="{field}"; filename="'.encode(), prepared.body)
            self.assertEqual(json.loads((Path(root) / "result.json").read_text()), world_result())
            with zipfile.ZipFile(Path(root) / "titration_test-job_results.zip") as archive:
                self.assertIn("04_targets/titration_targets.json", archive.namelist())
            self.assertFalse(list(Path(root).glob("*.part")))
            for path in Path(root).glob("*.json"):
                self.assertNotIn("test-only-token", path.read_text())

    def test_failed_interrupted_and_unknown_states_never_fetch_result(self):
        for state in ("failed", "interrupted", "unexpected"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as root, patch("affordance.server.requests.post", return_value=response({"job_id": "test-job"})), patch("affordance.server.requests.get", return_value=response({"state": state})) as get:
                with self.assertRaises((RuntimeError, ValueError)):
                    LocalizationClient(self.config).locate_files(self.files, 0.001, "capture", Path(root))
                get.assert_called_once()

    def test_post_timeout_is_not_retried(self):
        with tempfile.TemporaryDirectory() as root, patch("affordance.server.requests.post", side_effect=requests.Timeout) as post:
            with self.assertRaisesRegex(RuntimeError, "check the server queue"):
                LocalizationClient(self.config).locate_files(self.files, 0.001, "capture", Path(root))
            post.assert_called_once()

    def test_poll_timeout_preserves_job_id(self):
        with tempfile.TemporaryDirectory() as root, patch("affordance.server.requests.post", return_value=response({"job_id": "test-job"})), patch("affordance.server.time.monotonic", side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                LocalizationClient(self.config).locate_files(self.files, 0.001, "capture", Path(root))
            self.assertEqual(json.loads((Path(root) / "job.json").read_text())["job_id"], "test-job")

    def test_get_retry_and_auth_failure(self):
        client = LocalizationClient(self.config)
        with patch("affordance.server.requests.get", side_effect=[response({}, 503), response({"state": "completed"})]) as get, patch("affordance.server.time.sleep"):
            self.assertEqual(client._get_json("http://localhost", {}, float("inf"))["state"], "completed")
            self.assertEqual(get.call_count, 2)
        with patch("affordance.server.requests.get", return_value=response({}, 401)) as get:
            with self.assertRaises(requests.HTTPError):
                client._get_json("http://localhost", {}, float("inf"))
            get.assert_called_once()

    def test_rejects_unsafe_job_id(self):
        with tempfile.TemporaryDirectory() as root, patch("affordance.server.requests.post", return_value=response({"job_id": "../escape"})), patch("affordance.server.requests.get") as get:
            with self.assertRaises(ValueError):
                LocalizationClient(self.config).locate_files(self.files, 0.001, "capture", Path(root))
            get.assert_not_called()

    def test_upload_formats_and_depth_precision(self):
        frame = sample_frame()
        decoded = cv2.imdecode(np.frombuffer(self.files["depth"][1], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_array_equal(decoded, frame.depth_u16)
        self.assertEqual(json.loads(self.files["camera"][1])["fx"], 100)
        np.testing.assert_allclose(json.loads(self.files["world_ee"][1])["T_world_ee"], np.eye(4))
        with self.assertRaises(ValueError):
            build_upload_files(replace(frame, depth_u16=frame.depth_u16.astype(np.uint8)), np.eye(4))
        with self.assertRaises(ValueError):
            build_upload_files(replace(frame, color_bgr=frame.color_bgr[:1]), np.eye(4))


if __name__ == "__main__":
    unittest.main()
