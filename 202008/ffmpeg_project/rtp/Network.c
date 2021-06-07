#include "Network.h"

int udpInit(UdpContext* udp) {
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

	int server_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in server_sockaddr;
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(MY_PORT);
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

	if (bind(server_sockfd, (struct sockaddr*)&server_sockaddr, sizeof(server_sockaddr)) == -1) {
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
}

int udpSend(const UdpContext* udp, const uint8_t* data, uint32_t len) {
	memset(buffer, 0, sizeof(buffer));
	int len = recv(conn, buffer, sizeof(buffer), 0);

	if (len <= 0) {
		printf("Server break flag %d.\n", len);
		break;
	}

	if (buffer[0] == 'e' && buffer[1] == 'x'
		&& buffer[2] == 'i' && buffer[3] == 't') {
		printf("Server break flag.\n");
		break;
	}
	//fputs(buffer, stdout);

	//TODO handle data
	if (len > BUFFER_SIZE) {
		len = BUFFER_SIZE;
	}
	int i = 0;

	for (i = 0; i < len; i++) {
		printf("%d ", buffer[i]);
	}
	printf("\n\n");
	for (i = 0; i < len; i++) {
		buffer[i] += 1;
		printf("%d ", buffer[i]);
	}
	send(conn, buffer, len, 0);
}

void finalizeInit(UdpContext* udp) {
	// finalize socket win32
	closesocket(server_sockfd);
	WSACleanup();
}