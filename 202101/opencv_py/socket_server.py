#!/usr/bin/python3
import socket
import ssl

class server_ssl:
    def build_listen(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain('cert/server.crt' ,'cert/server_rsa_private.pem.unsecure')
        ssl.match_hostname('cert/server.crt' ,'127.0.0.1');
        with socket.socket(socket.AF_INET ,socket.SOCK_STREAM ,0) as sock:
            sock.bind(('127.0.0.1' ,9443))
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
    netstat -ant | findStr "LISTEN"

'''
        
if __name__ == "__main__":
    server = server_ssl()
    server.build_listen()
