#!/usr/bin/python3
import socket
import ssl

class server_ssl:
    def build_listen(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('cert/server.crt' ,'cert/server_rsa_private.pem.unsecure')
        with socket.socket(socket.AF_INET ,socket.SOCK_STREAM ,0) as sock:
            sock.bind(('127.0.0.1' ,55222))
            sock.listen(5)
            with context.wrap_socket(sock ,server_side = True) as ssock:
                while True:
                    client_socket ,addr = ssock.accept()
                    msg = client_socket.recv(1024).decode("utf-8")
                    print(f"receive msg from client {addr}:{msg}")
                    msg = f"Yes ,you got ssl content with server.\r\n".encode("utf-8")
                    client_socket.send(msg)
                    client_socket.close()

'''
    ssl 
'''
        
if __name__ == "__main__":
    #win10 netstat -ano | findstr55222
    server = server_ssl()
    server.build_listen()
