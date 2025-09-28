import threading

# Globaler State (einfach)
_server_instance = None
_server_thread = None
_server_url = None


def start_webview_once(root_dir=".", open_rel_path="/oedo-viewer/pages/generated/generated.html", port=None,
                       new_window=False):
    """
    Startet webview.serve_and_open() nur, wenn noch kein Server läuft.
    Gibt (server, thread, url) zurück.
    """
    global _server_instance, _server_thread, _server_url

    if _server_instance is not None and _server_thread is not None and _server_thread.is_alive():
        print(f"[INFO] Server läuft bereits unter {_server_url}")
        return _server_instance, _server_thread, _server_url

    # Neuer Start
    from handler import webview
    server, thread, url = webview.serve_and_open(
        root_dir=root_dir,
        open_rel_path=open_rel_path,
        port=port,
        new_window=new_window
    )

    _server_instance = server
    _server_thread = thread
    _server_url = url
    print(f"[INFO] Neuer Server gestartet unter {url}")
    return server, thread, url


server, thread, url = start_webview_once()