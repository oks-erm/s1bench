import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import local_runtime as runtime
import benchmark


class ProfileTests(unittest.TestCase):
    def test_mac_example_matches_managed_runtimes_without_enabling_calls(self):
        cfg = benchmark.read_config(Path(__file__).resolve().parents[1] / "config.mac.example.json")
        self.assertFalse(any(m.get("enabled") for m in cfg["models"]))
        self.assertFalse(any(m.get("api_key") or m.get("headers") for m in cfg["models"]))
        with patch.object(runtime.sys, "platform", "darwin"), patch.object(runtime.Path, "exists", return_value=True):
            self.assertEqual([m["name"] for m in cfg["models"] if runtime.can_manage(m)],
                             ["laya", "nimble", "clm"])


@unittest.skipUnless(os.name == "posix", "Local Mac process manager needs POSIX locks")
class RuntimeTests(unittest.TestCase):
    def test_owned_process_is_stopped_even_when_inference_fails(self):
        response = MagicMock()
        response.__enter__.return_value.status = 200
        process = MagicMock()
        process.poll.return_value = None
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(runtime, "can_manage", return_value=True),
            patch.object(runtime.local_models, "LOCAL", Path(root)),
            patch.object(runtime.local_models, "listening", return_value=False),
            patch.object(runtime, "urlopen", return_value=response),
            patch.object(runtime.subprocess, "Popen", return_value=process) as launch,
            patch.object(runtime, "stop_process") as stop,
        ):
            with self.assertRaisesRegex(ValueError, "inference failed"):
                with runtime.model_session({"name": "laya"}, threading.Event()):
                    raise ValueError("inference failed")
            launch.assert_called_once()
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            stop.assert_called_once_with(process)

    def test_existing_server_is_never_adopted_or_stopped(self):
        with (
            tempfile.TemporaryDirectory() as root,
            patch.object(runtime, "can_manage", return_value=True),
            patch.object(runtime.local_models, "LOCAL", Path(root)),
            patch.object(runtime.local_models, "listening", side_effect=lambda port: port == 8000),
            patch.object(runtime.subprocess, "Popen") as launch,
            patch.object(runtime, "stop_process") as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "separately running"):
                with runtime.model_session({"name": "laya"}, threading.Event()):
                    self.fail("Must refuse a running server")
            launch.assert_not_called()
            stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
