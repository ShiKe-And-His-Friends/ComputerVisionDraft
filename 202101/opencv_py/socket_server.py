#!/usr/bin/python3
import socket

class server_class:
    def build_listen(self):
        server_socket = socket.socket(socket.AF_INET ,socket.SOCK_STREAM)
        server_socket.bind(('127.0.0.1' ,55222))
        server_socket.listen(5)

        while True:
            client_socket ,addr = server_socket.accept()
            msg = client_socket.recv(1024).decode("utf-8")
            print(f"receive msg from client {addr}:{msg}")

            msg = f"Yes,I get it.\r\n".encode("utf-8")
            client_socket.send(msg)
            client_socket.close()

if __name__ == "__main__":
    server = server_class()
    server.build_listen()
