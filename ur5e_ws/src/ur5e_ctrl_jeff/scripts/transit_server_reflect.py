#!/usr/bin/env python

import requests
from http.server import BaseHTTPRequestHandler, HTTPServer

class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Handles HTTP requests and forwards them to the target server.
    """
    
    target_host = "172.16.78.10"    # IP address of the remote server
    target_port = 34389             # Port of the remote server

    def do_POST(self):
        """
        Handles POST requests by forwarding them to the target server and returning the response.
        """
        
        # Read the content length and request data
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        # Construct the target URL
        target_url = f"http://{self.target_host}:{self.target_port}{self.path}"
        
        try:

            print(f"[INFO] Forwarding request to {target_url}")

            # Forward the request to the target server
            response = requests.post(target_url, data=post_data, headers=self.headers, timeout=1000*60)
            
            # Send the response back to the client
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.content)

        except Exception as e:
            print(f"[ERROR] Error while forwarding request: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {e}".encode('utf-8'))

    def log_message(self, format, *args):
        """
        Suppresses default logging.
        """
        return

class ProxyServer:
    """
    Sets up and starts the proxy server.
    """

    def __init__(self, proxy_host, proxy_port):
        """
        Initializes the proxy server with the given host and port.

        :param proxy_host: Host address for the proxy server
        :param proxy_port: Port for the proxy server
        """
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

    def start(self):
        """
        Starts the proxy server and begins listening for incoming requests.
        """
        server_address = (self.proxy_host, self.proxy_port)
        httpd = HTTPServer(server_address, ProxyHTTPRequestHandler)
        print(f"[INFO] Proxy server listening on {self.proxy_host}:{self.proxy_port}")
        httpd.serve_forever()

if __name__ == "__main__":
    
    # Set the host and port for the proxy server
    proxy_host = "192.168.2.3"  # Replace with the IP address of computer2
    proxy_port = 5050           # Replace with the desired port for the proxy server
    
    # Initialize and start the proxy server
    proxy_server = ProxyServer(proxy_host, proxy_port)
    proxy_server.start()
    