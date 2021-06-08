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
		printf("Input Format Error!\n %s input_h264_file" ,argv[0]);
		return 0;
	}
	fileName = argv[1];

	printf("Rtp program start.\n");

	RtpContext rtpContext;
	UdpContext udpContext = {
		.dstIp = "127.0.0.1",
		.dstPort = "4483"
	};

	result = readFile(&stream ,&length ,fileName);
	if (result) {
		printf("Read file Error\n");
		return -1;
	}
	const char* name = "C\:\\Users\\Dcfr-186\\Videos\\Captures\\sample_copy.h264";
	FILE* fp = NULL;
	fp = fopen(name, "wb+");
	fwrite(stream ,1 ,length,fp);
	fclose(fp);
	printf("file rewrite %d\n" ,length);

	free(stream);
	printf("Rtp program stop.\n");
	return 0;
}