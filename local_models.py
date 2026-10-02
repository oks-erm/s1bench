"""Launch one installed model at a time on this 16 GB Mac."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
from urllib.error import URLError
from urllib.request import urlopen

import benchmark

BASE = Path(__file__).resolve().parent
LOCAL = BASE / ".venv/local"
PORTS = {"laya": 8000, "nimble": 11435, "clm": 8700}


def listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def command_for(model):
    env = os.environ.copy()
    env.update(HF_HOME=str(LOCAL / "huggingface"), HF_HUB_DISABLE_XET="1",
               HF_HUB_DOWNLOAD_TIMEOUT="60", PYTHONUNBUFFERED="1")
    if model == "laya":
        env.update(LAYA_HOST="127.0.0.1", LAYA_PORT="8000", LAYA_DEVICE="mps",
                   LAYA_MODELS="english", LAYA_REVISION="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
        command = [str(LOCAL / "laya/bin/laya-serve")]
    elif model == "nimble":
        env.update(OLLAMA_HOST="127.0.0.1:11435", OLLAMA_MODELS=str(LOCAL / "ollama-models"),
                   OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_NUM_PARALLEL="1", OLLAMA_NO_CLOUD="1")
        command = [str(LOCAL / "ollama/ollama"), "serve"]
    else:
        command = [str(LOCAL / "clm/bin/python"), str(BASE / "local_clm_server.py")]
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=PORTS)
    parser.add_argument("--dry-run", action="store_true", help="Show launch command without starting or editing anything")
    args = parser.parse_args()
    command, env = command_for(args.model)
    if args.dry_run:
        print(json.dumps(command))
        return 0
    if not Path(command[0]).exists():
        parser.error("Local runtime is missing. See LOCAL_SETUP.md.")
    # Hold the lock through model startup, before its listening socket exists.
    import fcntl
    with (LOCAL / "active-model.lock").open("a") as model_lock:
        try:
            fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another model launcher is active. Stop it with Ctrl+C first.")
        active = [name for name, port in PORTS.items() if listening(port)]
        if active:
            parser.error("Stop the current model server first (Ctrl+C in its terminal): " + ", ".join(active))
        if listening(11434):
            try:
                with urlopen("http://127.0.0.1:11434/api/ps", timeout=3) as response:
                    loaded = json.load(response).get("models", [])
            except (OSError, ValueError, URLError) as exc:
                parser.error("Cannot check memory use of your existing Ollama: " + str(exc))
            if loaded:
                parser.error("Unload the model in your existing Ollama before starting another large model.")
        config_path, _ = benchmark.initialize()
        cfg = benchmark.read_config(config_path)
        if args.model not in {m["name"] for m in cfg["models"]}:
            parser.error("The selected model profile is missing from config.json.")
        if not listening(8501):
            with (LOCAL / "dashboard.log").open("a") as log:
                subprocess.Popen([str(BASE / ".venv/bin/python"), "-m", "streamlit", "run",
                                  str(BASE / "bench_ui.py"), "--server.address", "127.0.0.1",
                                  "--server.port", "8501", "--server.headless", "true",
                                  "--browser.gatherUsageStats", "false"], cwd=BASE,
                                 stdout=log, stderr=log, start_new_session=True)
        print("Dashboard: http://127.0.0.1:8501", flush=True)
        print("Standalone server: wait for readiness before making manual HTTP or CLI requests.", flush=True)
        print("Stop this server before using the dashboard's automatic Run or Test buttons.", flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop this model before choosing another.", flush=True)
        try:
            return subprocess.call(command, cwd=BASE, env=env)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
