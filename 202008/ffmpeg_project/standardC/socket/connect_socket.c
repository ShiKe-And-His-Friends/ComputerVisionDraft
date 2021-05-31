#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <WinSock2.h>

#pragma  comment(lib,"ws2_32.lib")

#define MY_PORT 4428
#define QUEUE 20
#define BUFFER_SIZE 1024

//netstat - na | findstr "4482"

int main() {
	int server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in server_sockaddr;
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(MY_PORT);
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

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

	char buffer[BUFFER_SIZE];
	struct sockaddr_in client_addr;
	int length = sizeof(client_addr);

	int conn = accept(server_sockfd, (struct sockaddr*)&client_addr, &length);
	if (conn < 0) {
		printf("Server accept failure.\n");
		exit(-1);
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

	return 0;
}