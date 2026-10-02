"""Local-only HTTP adapter for the approved CLM 8-bit MLX community port."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
import sys
import time

MODEL_ID = "mlx-community/CLM-v0.1-8B-MLX-8bit"
MODEL_DIR = Path(__file__).resolve().parent / ".venv/local/clm-model"
MAX_BODY = 65536
ALLOCATOR_CACHE_BYTES = 256 * 1024 * 1024


def validate_request(body):
    if not isinstance(body, dict) or body.get("state") is None:
        raise ValueError("A JSON object with a non-null state is required.")
    if body.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError("Unknown model; use " + MODEL_ID)
    questions = body.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError("questions must contain between 1 and 64 named questions.")
    for question in questions.values():
        if not isinstance(question, dict):
            raise ValueError("Every question must be an object.")
        kind, criteria = question.get("type"), question.get("criteria")
        if kind not in ("choice", "noul", "score"):
            raise ValueError("Question type must be choice, noul, or score.")
        if kind == "choice" and (not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26):
            raise ValueError("choice criteria must have 2–26 options.")
        if kind == "score" and (not isinstance(criteria, list) or not 2 <= len(criteria) <= 10):
            raise ValueError("score criteria must have 2–10 levels.")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise ValueError("noul criteria must be an object when supplied.")
    return body["state"], questions


def create_server(engine, port=8700):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, status, body, elapsed=None):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if elapsed is not None:
                self.send_header("X-CLM-Latency-Ms", str(elapsed))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ok", "model": MODEL_ID, "device": "mlx",
                                 "max_tokens": 2048, "cache_size": 4096,
                                 "allocator_cache_bytes": ALLOCATOR_CACHE_BYTES})
            elif self.path == "/v1/models":
                self.reply(200, {"data": [{"id": MODEL_ID}]})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.reply(400, {"error": "Use Content-Length, not chunked encoding."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    self.reply(413, {"error": "Request must contain 1–65536 bytes."})
                    return
                state, questions = validate_request(json.loads(self.rfile.read(size)))
            except (ValueError, UnicodeError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            started = time.perf_counter()
            try:
                result = engine.answer(state, questions)
                result["model"] = MODEL_ID
                self.reply(200, result, (time.perf_counter() - started) * 1000)
            except ValueError as exc:
                self.reply(422, {"error": str(exc)})
            except Exception:
                logging.exception("CLM inference failed")
                self.reply(500, {"error": "CLM inference failed; inspect the local server log."})

    # Single-threaded: MLX initialization and inference stay on the same thread.
    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    sys.path.insert(0, str(MODEL_DIR))
    import mlx.core as mx
    # MLX defaults this cache to 1.5 times the device working set, too large for
    # a long run alongside the dashboard on a 16 GB Mac. This caches free buffers,
    # independently of the model's projection cache, and does not change weights.
    mx.set_cache_limit(ALLOCATOR_CACHE_BYTES)
    from clm_mlx.engine import Engine
    print("Loading CLM 8-bit MLX; keep other large models stopped.", flush=True)
    engine = Engine(str(MODEL_DIR / "encoder"), str(MODEL_DIR / "heads"),
                    max_tokens=2048, cache_size=4096)
    engine.encoder.batch_tokens = 2048
    with create_server(engine) as server:
        print("CLM ready at http://127.0.0.1:8700", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
