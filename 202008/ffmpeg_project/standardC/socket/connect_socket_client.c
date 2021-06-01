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

	int sock_cli = socket(AF_INET, SOCK_STREAM, 0);

	struct sockaddr_in servaddr;
	memset(&servaddr, 0, sizeof(struct sockaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(MY_PORT);
	servaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	if (connect(sock_cli, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
		printf("client connect failure.\n");
		exit(-1);
	}
	else {
		printf("client connect success.\n");
	}
	char sendbuf[BUFFER_SIZE] = "safewwe2145";
	char recvbuf[BUFFER_SIZE];
	while (fgets(sendbuf ,sizeof(sendbuf) ,stdin) != NULL) {
		//TODO handle send data
		send(sock_cli ,sendbuf ,strlen(sendbuf) ,0);
		if (strcmp(sendbuf ,"exit\n") == 0) {
			printf("client connect close.\n");
			break;
		}
		//TODO handle receive data
		recv(sock_cli ,recvbuf ,sizeof(recvbuf) ,0);
		fputs(recvbuf ,stdout);

		memset(sendbuf ,0 ,sizeof(sendbuf));
		memset(recvbuf, 0, sizeof(recvbuf));
	}
#ifdef WIN_32_SOCK_H
	// finalize socket win32
	closesocket(sock_cli);
	WSACleanup();
#endif
	printf("client realsea sources.\n");
	return 0;
}