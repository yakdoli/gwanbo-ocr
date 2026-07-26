from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import ClassVar
from urllib.error import HTTPError

import pytest

from gwanbo_ocr.pdf.openai_review import UrlLibJsonSender


class RedirectHandler(BaseHTTPRequestHandler):
    visited_sink: ClassVar[bool] = False

    def do_POST(self) -> None:
        self.send_response(302)
        self.send_header("Location", "/sink")
        self.end_headers()

    def do_GET(self) -> None:
        type(self).visited_sink = True
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"redirected")

    def log_message(self, format: str, *args: str) -> None:
        del format, args
        return


def test_url_sender_rejects_redirects_before_forwarding_credentials() -> None:
    RedirectHandler.visited_sink = False
    server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with pytest.raises(HTTPError):
            UrlLibJsonSender().post(
                f"http://127.0.0.1:{server.server_port}/start",
                {"Authorization": "Bearer secret"},
                b"{}",
                1,
            )
        assert RedirectHandler.visited_sink is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
