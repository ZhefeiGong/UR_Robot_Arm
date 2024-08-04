import socket
import threading
import requests
import json
import time
import traceback

class ProxyServer:
    """
    Proxy server to forward requests from Client1 to Target Server
    """

    def __init__(self, proxy_host, proxy_port, target_host, target_port):
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.target_host = target_host
        self.target_port = target_port

    def handle_client(self, client_socket):
        try:
            request = client_socket.recv(4096)
            if not request:
                return

            # Assume request is a JSON string with at least 'endpoint' and 'data' fields
            request_data = json.loads(request.decode('utf-8'))
            endpoint = request_data.get("endpoint")
            data = request_data.get("data")

            if endpoint == "/inference":
                response = self.forward_request("/inference", data)
            elif endpoint == "/load":
                response = self.forward_request("/load", data)
            elif endpoint == "/check":
                response = self.forward_request("/check", data)
            else:
                response = {"error": "Unknown endpoint"}

            client_socket.send(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            response = {"error": str(e)}
            client_socket.send(json.dumps(response).encode('utf-8'))

        finally:
            client_socket.close()

    def forward_request(self, endpoint, data):
        url = f"http://{self.target_host}:{self.target_port}{endpoint}"

        try:
            response = requests.post(url, data=json.dumps(data), timeout=1000*60)
            return response.json()
        
        except Exception as e:
            print(f"[ERROR] forwarding request: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def start(self):

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.proxy_host, self.proxy_port))
        server.listen(5)
        print(f"[INFO] Proxy server listening on {self.proxy_host}:{self.proxy_port}")

        while True:
            client_socket, addr = server.accept()
            print(f"[INFO] Accepted connection from {addr}")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler.start()


if __name__ == "__main__":

    proxy_host = "192.168.0.9"  # Listening on all interfaces
    proxy_port = 5050       # Proxy port on computer2
    
    target_host = "172.16.78.10"  # Target server's host
    target_port = 36095        # Target server's port

    proxy_server = ProxyServer(proxy_host, proxy_port, target_host, target_port)
    proxy_server.start()
