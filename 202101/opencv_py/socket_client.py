#!/usr/bin/python3
import socket
import ssl

class client_ssl:
    def send_hello(self,):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations('cert/ca.crt')

        with socket.create_connection(('127.0.0.1' ,9443)) as sock:
            with context.wrap_socket(sock ,server_hostname = '127.0.0.1') as ssock:
                msg = "Do i connect ssl with server?".encode("utf-8")
                ssock.send(msg)
                msg = ssock.recv(1024).decode("utf-8")
                print(f"receive msg from server:{msg}")
                ssock.close()

if __name__ == "__main__":
    client = client_ssl()
    client.send_hello()
