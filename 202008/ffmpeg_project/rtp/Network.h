#include <stdio.h>
#include "typedef.h"

#ifndef RTP_NETWORK_H
#define RTP_NETWORK_H

#define WIN_32_SOCK_H
#ifdef WIN_32_SOCK_H

#include <WinSock2.h>
#pragma  comment(lib,"ws2_32.lib")

#endif

typedef struct {
	const char* dstIp;
	int dstPort;
	struct sockaddr_in addr;
	int socket;
}UdpContext;

int udpInit(UdpContext *udp);

/* send UDP packet */
int udpSend(const UdpContext *udp ,const uint8_t *data ,uint32_t len);

void finalizeInit(UdpContext* udp);

#endif
