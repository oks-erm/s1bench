"""Own local server lifetimes during a sequential comparison on this Mac."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

import local_models
from local_clm_server import ALLOCATOR_CACHE_BYTES

MODEL_IDS = {"laya": "english", "nimble": "nimble",
             "clm": "mlx-community/CLM-v0.1-8B-MLX-8bit"}


def can_manage(model):
    name = model.get("name")
    if sys.platform != "darwin" or name not in MODEL_IDS:
        return False
    endpoint = f"http://127.0.0.1:{local_models.PORTS[name]}/v1/systemone"
    command, _ = local_models.command_for(name)
    return (model.get("model") == MODEL_IDS[name] and model.get("endpoint") == endpoint
            and model.get("api") == "systemone" and Path(command[0]).exists())


def stop_process(process):
    # The process group was created by this context; never stop a borrowed server.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextmanager
def model_session(model, cancel):
    if not can_manage(model):
        yield
        return
    import fcntl
    name = model["name"]
    if name == "clm":
        note = f"MLX free-buffer allocator cache limit: {ALLOCATOR_CACHE_BYTES} bytes."
        previous = model.get("notes") or ""
        if note not in previous:
            model["notes"] = (previous + " " + note).strip()
    with (local_models.LOCAL / "active-model.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Stop the separately launched local model before starting this comparison.") from exc
        active = [alias for alias, port in local_models.PORTS.items() if local_models.listening(port)]
        if active:
            raise RuntimeError("Stop separately running model servers first: " + ", ".join(active))
        if local_models.listening(11434):
            with urlopen("http://127.0.0.1:11434/api/ps", timeout=3) as response:
                if json.load(response).get("models"):
                    raise RuntimeError("Unload the model in the existing Ollama app before this comparison.")
        command, env = local_models.command_for(name)
        for key in ("OPENAI_API_KEY", "JEV_API_KEY"):
            env.pop(key, None)
        with (local_models.LOCAL / (name + "-benchmark.log")).open("a") as log:
            process = subprocess.Popen(command, cwd=local_models.BASE, env=env,
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                path = "/api/version" if name == "nimble" else "/health"
                url = f"http://127.0.0.1:{local_models.PORTS[name]}{path}"
                deadline = time.monotonic() + 300
                while not cancel.is_set():
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited during startup; inspect its local benchmark log.")
                    try:
                        with urlopen(url, timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{name} did not become ready within five minutes.")
                    cancel.wait(.25)
                yield
            finally:
                stop_process(process)
