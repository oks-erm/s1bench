import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import local_models


@unittest.skipUnless(sys.platform == "darwin", "Mac-specific local launcher")
class LocalLauncherTests(unittest.TestCase):
    def test_starting_server_preserves_saved_run_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            original = json.dumps({"models": [
                {"name": "laya", "enabled": False},
                {"name": "clm", "enabled": True},
            ]})
            config.write_text(original)
            with (patch.object(sys, "argv", ["local_models.py", "laya"]),
                  patch.object(local_models, "LOCAL", root),
                  patch.object(local_models, "command_for", return_value=([sys.executable], {})),
                  patch.object(local_models, "listening", side_effect=lambda port: port == 8501),
                  patch.object(local_models.benchmark, "initialize", return_value=(config, None)),
                  patch.object(local_models.benchmark, "read_config", return_value=json.loads(original)),
                  patch.object(local_models.subprocess, "call", return_value=0) as launch):
                self.assertEqual(local_models.main(), 0)
            launch.assert_called_once()
            self.assertEqual(config.read_text(), original)
            self.assertFalse(config.with_suffix(".json.bak").exists())


if __name__ == "__main__":
    unittest.main()
