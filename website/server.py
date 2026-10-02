"""
Pelid Local Website Preview Server
Serves the product website and interactive calculator locally on http://localhost:3000
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from pathlib import Path

# Force UTF-8 encoding
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORT = 3000
WEB_DIR = Path(__file__).resolve().parent

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Clean custom request log
        sys.stdout.write(f"  [HTTP] {self.address_string()} - {format % args}\n")

def main():
    os.chdir(WEB_DIR)
    
    # Try preferred port 3000, fallback to 3001 if in use
    port = PORT
    server = None
    for p in range(3000, 3010):
        try:
            server = socketserver.TCPServer(("", p), CustomHandler)
            port = p
            break
        except OSError:
            continue

    if not server:
        print("[ERROR] Could not bind to ports 3000-3010.")
        sys.exit(1)

    url = f"http://localhost:{port}"
    print("=" * 65)
    print("  PELID PRODUCT WEBSITE & INTERACTIVE ROI CALCULATOR")
    print("=" * 65)
    print(f"  Local URL:        {url}")
    print(f"  Directory:        {WEB_DIR}")
    print("  Status:           Ready for local evaluation ($0.00 cost)")
    print("-" * 65)
    print("  Press Ctrl+C to stop the preview server.")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [SHUTDOWN] Preview server stopped cleanly.")
        server.server_close()

if __name__ == "__main__":
    main()
