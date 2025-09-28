import os
import socket
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial


def _find_free_port(start=8100, limit=200):
    """Find a free TCP port starting from 'start'."""
    for p in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found.")


def serve_and_open(root_dir=".", open_rel_path="generated.html", port=None, new_window=True):
    """
    Start a blocking local HTTP server and open the page in the browser.

    Args:
        root_dir (str): Directory to serve as document root.
        open_rel_path (str): Path relative to root_dir to open in browser.
        port (int|None): Fixed port; if None => auto-find free port.
        new_window (bool): Open in a new window (if browser allows).
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise ValueError(f"Directory does not exist: {root_dir}")

    if port is None:
        port = _find_free_port(8100)

    handler = partial(SimpleHTTPRequestHandler, directory=root_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)

    url = f"http://127.0.0.1:{port}/{open_rel_path.lstrip('/')}"
    if new_window:
        webbrowser.open_new(url)
    else:
        webbrowser.open(url, new=0)

    print(f"[OK] Server running at: {url}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down server...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    serve_and_open(
        root_dir="../",
        open_rel_path="oedo-viewer/pages/generated/generated.html",
        port=8080,
        new_window=False,
    )