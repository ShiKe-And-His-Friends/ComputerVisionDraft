/**
* Thanks for https://github.com/hmgle/h264_to_rtp
* Copy from it.
*/

#include <stdio.h>
#include <stdlib.h>
#include "send_h264_file_rtp.h"

#define DEFAULT_DEST_PORT 1234
#define NAL_BUF_SIZE 1500 * 50

uint16_t DEST_PORT;
linklist CLIENT_IP_LIST;
uint8_t nal_buf[NAL_BUF_SIZE];

static void add_client_list(linklist client_ip_list ,char* ipaddr) {
	struct sockaddr_in server_c;
	pnode pnode_tmp;
	const int on = 1;

	insert_nodulp_node();
}

static int h264nal2rtp_send(int frameRate ,uint8_t* pstStream ,int nalu_len ,linklist client_ip_list) {
	
}

void add_client(linklist client_ip_list ,char*  ipaddr) {
	
}

static int copy_nal_from_file(FILE* p, uint8_t* buf, int* len) {
	char tmpbuf[4];
	char tmpbuf2[1];
	int flag = 0;
	int ret;

#if 0
	ret = fread(tmpbuf ,4 ,1 fp);
	if (!ret) {
		return 0;
	}
#endif
	do {
	} while (1);

}

int main(int argc ,char** argv) {
	FILE* fp;
	FILE* fp_test;
	int len;
	int ret;

	if (argc < 3) {
		fprintf(stderr ,"usage : %s <input_file> <dstip> [:dest_port]\n" ,argv[0]);
	}
	fp = fopen(argv[1] ,"r");
	if (!fp) {
		perror("fopen");
		exit(errno);
	}
	fp_test = fopen("file_test.h264" ,"w");
	if (!fp_test) {
		perror("fopen");
		exit(errno);
	}
	if (argc > 3) {
		DEST_PORT = atoi(argv[3]);
	}
	else {
		DEST_PORT = DEFAULT_DEST_PORT;
	}

	CLIENT_IP_LIST = create_null_list_link();
	add_client_list(CLIENT_IP_LIST ,argv[2]);

	fprintf(stderr ,"DEST_PORT is %d \n" ,DEST_PORT);

	while (copy_nal_from_file(fp ,nal_buf ,&len) != -1) {
		ret = h264nal2rtp_send(25 ,nal_buf ,len ,CLIENT_IP_LIST);
		if (ret != -1){
			Sleep(20);
		}
	}

	//debug_print();

	fclose(fp_test);
	fclose(fp);

	return 0;
}