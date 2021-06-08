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
		return -1;
	}

	if (LOBYTE(wsaData.wVersion) != 1 ||
		HIBYTE(wsaData.wVersion) != 1) {
		WSACleanup();
		printf("socket init failure 2.\n");
		return -1;
	}
	printf("socket init success.\n");

	if (udp == NULL || udp->dstIp == NULL || udp->dstPort == 0) {
		printf("socket udp not init.\n");
		return -1;
	}
	udp->socket = socket(AF_INET, SOCK_DGRAM, 0);
	if (udp->socket < 0) {
		printf("socket error.\n");
		return -1;
	}
	udp->addr.sin_family = AF_INET;
	udp->addr.sin_port = htons(udp->dstPort);
	// udp->addr.sin_addr.s_addr = htonl(udp->dstIp);
	udp->addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

	int num = (int)sendto(udp->socket ,"" ,1 ,0 ,(struct sockaddr *)&udp->addr ,sizeof(udp->addr));
	if (num != 1) {
		int errorNo = WSAGetLastError();
		printf("udp init sendto test err, %d  %d\n" ,num ,errorNo);
		return -1;
	}
	printf("socket test success.\n");
	return 0;
}

int udpSend(const UdpContext* udp, const uint8_t* data, uint32_t len) {
	if (udp == NULL || udp->dstIp == NULL) {
		printf("socket send failure.\n");
		return -1;
	}
	int num = (int)sendto(udp->socket, data, len, 0, (struct sockaddr*)&udp->addr, sizeof(udp->addr));
	if (num != len) {
		printf("%s sendto err. %d %d\n", udp->dstIp ,num ,len);
		return -1;
	}
	return len;
}

void udpFinalize(UdpContext* udp) {
	// finalize socket win32
	closesocket(udp->socket);
	WSACleanup();
	printf("udp finalize success.\n");
}