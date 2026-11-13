"""The live page's server: page headers, the viewer port, and the throw queue."""

import socket
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from view.server import LiveView


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.port = free_port()
        self.view = LiveView(self.port, viewer_port=9001)
        self.addCleanup(self.view.close)

    def post(self, path: str):
        return urlopen(Request(f"http://127.0.0.1:{self.port}{path}", method="POST"))

    def test_page_carries_viewer_port_and_isolation_headers(self) -> None:
        r = urlopen(f"http://127.0.0.1:{self.port}/")
        self.assertEqual(r.headers["Cross-Origin-Opener-Policy"], "same-origin")
        self.assertEqual(r.headers["Cross-Origin-Embedder-Policy"], "require-corp")
        body = r.read().decode()
        self.assertNotIn("__VIEWER_PORT__", body)
        self.assertIn("9001", body)

    def test_page_without_a_viewer_gets_port_zero(self) -> None:
        port = free_port()
        view = LiveView(port)
        self.addCleanup(view.close)
        body = urlopen(f"http://127.0.0.1:{port}/").read().decode()
        self.assertIn("const VIEWER_PORT = 0;", body)

    def test_throws_queue_in_order_and_drain_once(self) -> None:
        for aim in ("gripper", "forearm", "gripper"):
            self.assertEqual(self.post(f"/throw?aim={aim}").status, 204)
        self.assertEqual(self.view.take_throws(), ["gripper", "forearm", "gripper"])
        self.assertEqual(self.view.take_throws(), [])

    def test_bad_aim_is_rejected(self) -> None:
        for path in ("/throw?aim=cube", "/throw"):
            with self.assertRaises(HTTPError) as caught:
                self.post(path)
            self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.view.take_throws(), [])


if __name__ == "__main__":
    unittest.main()
