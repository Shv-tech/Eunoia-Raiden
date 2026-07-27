"""
Eunoia Raiden — Dashboard Server
visualize/server.py

Serves the visualize/ folder over HTTP (so the dashboard can fetch
data.json without browser file:// restrictions) and runs introspect.py
in a background thread, regenerating data.json whenever a source file
in core/ dsl/ engine/ factory/ training/ eval/ or the root scripts
changes. The dashboard polls data.json on its own, so it updates live
as you edit and save code.

Usage:
    python visualize/server.py
    python visualize/server.py --port 8765 --interval 2
"""

from __future__ import annotations

import argparse
import http.server
import functools
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

VIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VIS_DIR))

import introspect  # noqa: E402


def background_watch(interval: float):
    # introspect.watch() already performs an initial write on its first
    # loop iteration (last_hash starts as None), so we don't pre-write here.
    introspect.watch(interval)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default per-request spam; errors still raise normally


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between codebase re-scans")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    t = threading.Thread(target=background_watch, args=(args.interval,), daemon=True)
    t.start()

    handler = functools.partial(QuietHandler, directory=str(VIS_DIR))
    with ReusableTCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/dashboard.html"
        print("=" * 60)
        print("  Eunoia Raiden — Live Dashboard")
        print("=" * 60)
        print(f"  Serving  : {url}")
        print(f"  Rescans  : every {args.interval}s for source changes")
        print("  Stop     : Ctrl+C")
        print("=" * 60)
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] stopped.")


if __name__ == "__main__":
    main()