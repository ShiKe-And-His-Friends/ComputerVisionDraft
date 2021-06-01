#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define WIN_32_SOCK_H
#ifdef WIN_32_SOCK_H
	#include <WinSock2.h>
	#pragma  comment(lib,"ws2_32.lib")
#endif

#define MY_PORT 4483
#define QUEUE 20
#define BUFFER_SIZE 1024

//netstat -ano | findstr "4483"

int main() {

#ifdef WIN_32_SOCK_H
	// init socket win32
	WORD wVersionRequested;
	WSADATA wsaData;
	int err;
	wVersionRequested = MAKEWORD(1, 1);

	err = WSAStartup(wVersionRequested, &wsaData);
	if (err != 0) {
		printf("socket init failure 1.\n");
		return;
	}

	if (LOBYTE(wsaData.wVersion) != 1 ||
		HIBYTE(wsaData.wVersion) != 1) {
		WSACleanup();
		printf("socket init failure 2.\n");
		return;
	}
	printf("socket init success.\n");
#endif
	
	int server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in server_sockaddr;
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(MY_PORT);
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

	if (bind(server_sockfd, (struct sockaddr *)&server_sockaddr, sizeof(server_sockaddr)) == -1) {
		printf("Server bind port failure.\n");
		exit(-1);
	}
	else {
		printf("Server bind port success.\n");
	}
	if (listen(server_sockfd, QUEUE) == -1) {
		printf("Server listen failure.\n");
		exit(-1);
	}
	else {
		printf("Server listen success.\n");
	}

	char buffer[BUFFER_SIZE];
	struct sockaddr_in client_addr;
	int length = sizeof(client_addr);

	int conn = accept(server_sockfd, (struct sockaddr*)&client_addr, &length);
	if (conn < 0) {
		printf("Server accept failure.\n");
		exit(-1);
	}
	else {
		printf("Server accept success.\n");
	}

	while (1) {
		memset(buffer ,0 ,sizeof(buffer));
		int len = recv(conn ,buffer ,sizeof(buffer) ,0);
		if (strcmp(buffer ,"exit\n") == 0) {
			printf("Server break flag.\n");
			break;
			fputs(buffer ,stdout);
			send(conn ,buffer ,len ,0);
		}
	}

	close(conn);
	close(server_sockfd);
#ifdef WIN_32_SOCK_H
	// finalize socket win32
	closesocket(server_sockfd);
	WSACleanup();
#endif
	printf("Server realsea sources.\n");
	return 0;
}