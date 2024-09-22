#!/usr/bin/env python

import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi

class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    Handles HTTP requests and forwards them to the target server.
    """
    
    target_host = "172.16.78.10"    # IP address of the remote server
    target_port = 39017             # Port of the remote server

    def do_POST(self):
        """
        Handles POST requests by forwarding them to the target server and returning the response.
        """

        # Parse the content-type header to determine how to handle the POST data
        content_type, pdict = cgi.parse_header(self.headers.get('content-type'))

        # If the content is multipart/form-data, we need to handle files
        if content_type == 'multipart/form-data':
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            form_data = cgi.parse_multipart(self.rfile, pdict)

            # Extract JSON data and files from the form
            json_payload = form_data.get('json')[0]
            img_static = form_data.get('img_static')[0]
            img_gripper = form_data.get('img_gripper')[0]

            # Prepare the files and payload to forward
            files = {
                "json": json_payload,
                "img_static": ("img_stat.txt", img_static, "text/plain"),
                "img_gripper": ("img_grip.txt", img_gripper, "text/plain"),
            }
            
            # Construct the target URL
            target_url = f"http://{self.target_host}:{self.target_port}{self.path}"

            try:
                print(f"[INFO] Forwarding multipart request to {target_url}")

                # Forward the request to the target server
                response = requests.post(target_url, files=files, timeout=1000*60)
                
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

        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Unsupported Content-Type")

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
