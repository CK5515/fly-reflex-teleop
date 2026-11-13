"""A small web page that shows, live, who is steering the arm and why.

The teleop loop calls `publish()` with a snapshot a few times a second; the
page (view/index.html) listens on a server-sent-events stream. The page also embeds
MuJoCo's own web viewer (served by MuJoCo on `viewer_port`) and posts throw
requests here, which the teleop loop collects with `take_throws()`. Standard
library only.
"""

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PAGE = Path(__file__).with_name("index.html")
HISTORY = 400  # snapshots replayed to a page that connects late (20 s at 20 Hz)
AIMS = ("gripper", "forearm")


class LiveView:
    def __init__(self, port: int = 8000, viewer_port: int | None = None):
        self.port = port
        self._throws: list[str] = []
        self._throws_lock = threading.Lock()
        self._history: deque[tuple[int, str]] = deque(maxlen=HISTORY)
        self._seq = 0
        self._changed = threading.Condition()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep the teleop terminal quiet
                pass

            def do_GET(self) -> None:
                if self.path == "/events":
                    view._stream(self)
                elif urlsplit(self.path).path in ("/", "/index.html"):
                    body = PAGE.read_text().replace("__VIEWER_PORT__", str(viewer_port or 0)).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    # MuJoCo's viewer needs an isolated page to run inside a frame.
                    self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                    self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                url = urlsplit(self.path)
                aim = parse_qs(url.query).get("aim", [""])[0]
                if url.path != "/throw":
                    self.send_error(404)
                elif aim not in AIMS:
                    self.send_error(400, "aim must be gripper or forearm")
                else:
                    with view._throws_lock:
                        view._throws.append(aim)
                    self.send_response(204)
                    self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"live view at http://127.0.0.1:{port}")

    def take_throws(self) -> list[str]:
        """Throws asked for from the page since the last call, oldest first."""
        with self._throws_lock:
            throws, self._throws = self._throws, []
        return throws

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def publish(self, snapshot: dict) -> None:
        with self._changed:
            self._seq += 1
            self._history.append((self._seq, json.dumps(snapshot)))
            self._changed.notify_all()

    def _stream(self, request: BaseHTTPRequestHandler) -> None:
        request.send_response(200)
        request.send_header("Content-Type", "text/event-stream")
        request.send_header("Cache-Control", "no-store")
        request.end_headers()
        sent = 0
        try:
            while True:
                with self._changed:
                    self._changed.wait_for(lambda: self._seq > sent, timeout=5)
                    fresh = [(s, line) for s, line in self._history if s > sent]
                if not fresh:
                    request.wfile.write(b": still here\n\n")  # keeps the connection open
                for seq, line in fresh:
                    request.wfile.write(f"data: {line}\n\n".encode())
                    sent = seq
                request.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
