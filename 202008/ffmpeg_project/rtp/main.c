/*
 Thanks for Github: https://github.com/lmshao/RTP
 Created by Liming Shao on 2018/5/11.
 Copy And Use In My Project.
*/

#include <stdio.h>
#include <string.h>
#include "RtpEnc.h"
#include "Utils.h"

int main(int argc ,char** argv) {
	uint8_t* stream = NULL;
	uint32_t length;
	uint32_t result;
	char* fileName = "sugar.h264";
	if (argc != 2) {
		printf("input format error!\n %s input_h264_file" ,argv[0]);
		return 0;
	}
	fileName = argv[1];

	printf("rtp program start.\n");

	RtpContext rtpContext;
	UdpContext udpContext = {
		.dstIp = "127.0.0.1",
		.dstPort = 4483
	};

	result = readFile(&stream ,&length ,fileName);
	if (result) {
		printf("read file error\n");
		return -1;
	}
	else {
		printf("read file success\n");
	}

	result = udpInit(&udpContext);
	if (result) {
		printf("udp init failure.\n");
		return -1;
	}
	else {
		printf("udp init success.\n");
	}

	initRtpContext(&rtpContext);

	rtpSend(&rtpContext ,&udpContext ,stream ,length);

	udpFinalize(&udpContext);
	free(stream);
	printf("rtp program stop.\n");
	return 0;
}