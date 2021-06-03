/*
 Thanks for Github: https://github.com/lmshao/RTP
 Created by Liming Shao on 2018/5/11.
 Copy And Use In My Project.
*/

#include <stdio.h>
#include <string.h>
#include "RtpEnc.h"
#include "Utils.h"

int main() {
	uint8_t* stream = NULL;
	uint32_t length;
	uint32_t result;
	const char* fileName = "sugar.h264";

	printf("Rtp program start.\n");

	RtpContext rtpContext;
	UdpContext udpContext = {
		.dstIp = "127.0.0.1",
		.dstPort = "4483"
	};

	result = readFile(&stream ,&length ,fileName);
	if (result) {
		printf("Read file Error\n");
		exit(-1);
	}

	printf("Rtp program stop.\n");
	return 0;
}