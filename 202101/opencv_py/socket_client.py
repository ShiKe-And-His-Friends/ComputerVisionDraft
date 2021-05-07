#!/usr/bin/python3
import socket

class client_class:
    def send_hello(self):
        client_socket = socket.socket(socket.AF_INET ,socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1' ,55222))

        msg = "did i connect with server?".encode("utf-8")
        client_socket.send(msg)
        
        msg = client_socket.recv(1024).decode('utf-8')
        print(f"receive msg from server:{msg}")
        client_socket.close()

if __name__ == "__main__":
    client = client_class()
    client.send_hello()
