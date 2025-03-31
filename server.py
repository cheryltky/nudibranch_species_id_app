"""
Simple HTTP Server for Nudibranch Species Identifier App

This script runs a simple HTTP server to serve the nudibranch identification app.
Run this script from the directory containing the app files.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path
import os

# Configuration
PORT = 8000
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for serving files from the current directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def run_server():
    """Run the HTTP server and open the app in a web browser."""

    # Check if the HTML file exists
    html_file = DIRECTORY / "nudibranch_identifier.html"
    if not html_file.exists():
        print(f"Error: {html_file} not found.")
        return

    # Start the server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        server_url = f"http://localhost:{PORT}/nudibranch_identifier.html"
        print(f"Server running at {server_url}")
        print("Press Ctrl+C to stop the server")

        # Open the app in the default web browser
        webbrowser.open(server_url)

        try:
            # Keep the server running
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run_server()
