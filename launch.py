"""
Catan Win-Probability Engine — Desktop Launcher

Works in two modes:
  • Frozen (PyInstaller EXE) — runs uvicorn in a daemon thread in-process.
    sys.executable is the EXE itself so spawning a subprocess is not viable.
  • Development (plain Python) — spawns uvicorn as a subprocess the normal way.
"""

import os
import socket
import sys
import time
import webbrowser

PORT = 8765
LOG_FILE = os.path.join(os.path.expanduser("~"), "catan_launch.log")

# PyInstaller sets sys.frozen = True and sys._MEIPASS = extraction dir
IS_FROZEN = getattr(sys, "frozen", False)


def _log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass
BASE_DIR = sys._MEIPASS if IS_FROZEN else os.path.dirname(os.path.abspath(__file__))  # type: ignore[attr-defined]
API_DIR = os.path.join(BASE_DIR, "api")


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_server(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _start_frozen_server(port: int) -> None:
    """Run uvicorn in-process as a daemon thread (EXE mode)."""
    import threading

    # Make the bundled packages importable
    sys.path.insert(0, BASE_DIR)  # catan_engine package lives here
    sys.path.insert(0, API_DIR)   # main.py + estimators/ live here

    import uvicorn  # noqa: PLC0415

    def _run():
        try:
            uvicorn.run("main:app", host="127.0.0.1", port=port, log_level="error", log_config=None)
        except Exception as exc:
            _log(f"uvicorn crashed: {exc}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def _start_dev_server(port: int):
    """Spawn uvicorn as a subprocess (development mode)."""
    import subprocess

    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # type: ignore[attr-defined]
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=API_DIR,
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    _log(f"=== Launch start IS_FROZEN={IS_FROZEN} BASE_DIR={BASE_DIR} ===")
    port = PORT
    if not port_free(port):
        for p in range(PORT + 1, PORT + 10):
            if port_free(p):
                port = p
                break
        else:
            # All ports occupied — a previous instance is likely running
            webbrowser.open(f"http://localhost:{PORT}/")
            return

    if IS_FROZEN:
        _log(f"Starting frozen server on port {port}")
        _start_frozen_server(port)
        _log("Thread started — waiting for server...")
        ok = wait_for_server(port)
        _log(f"Server ready={ok}")
        if ok:
            webbrowser.open(f"http://localhost:{port}/")
        else:
            _log("ERROR: server never became ready — check log above")
        # Keep the process alive while the daemon server thread runs
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
    else:
        proc = _start_dev_server(port)
        wait_for_server(port)
        webbrowser.open(f"http://localhost:{port}/")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        _log(f"FATAL: {exc}\n{traceback.format_exc()}")
