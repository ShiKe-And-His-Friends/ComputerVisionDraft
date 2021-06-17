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

void init_windows_socket(void) {
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
}

void finalize_windows_socket(void) {
	WSACleanup();
	printf("udp finalize success.\n");
}

static void add_client_list(linklist client_ip_list ,char* ipaddr) {
	struct sockaddr_in server_c;
	pnode pnode_tmp;
	const int on = 1;
	int ret = insert_nodulp_node(client_ip_list ,ipaddr);
	pnode_tmp = search_node(CLIENT_IP_LIST ,ipaddr);
	server_c.sin_family = AF_INET;
	server_c.sin_port = htons(DEST_PORT);
	//TODO windows/linux set ip
	//server_c.sin_addr.s_addr = inet_addr(ipaddr);
	server_c.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
	pnode_tmp->sned_fail_n = 0;
	pnode_tmp->node_info.socket_c = socket(AF_INET ,SOCK_DGRAM ,0);

	if (setsockopt(pnode_tmp->node_info.socket_c ,SOL_SOCKET ,SO_BROADCAST ,&on ,sizeof(on) < 0)){
		fprintf(stderr ,"initSvr:Socket options set error.\n");
		//TODO check loop
		//exit(errno);
	}

	if ((connect(pnode_tmp->node_info.socket_c ,(const struct sockaddr *)&server_c ,sizeof(struct sockaddr_in))) == -1) {
		perror("connect");
		//TODO check tcp
		//exit(-1);
	}
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
	fp = fopen(argv[1] ,"rb");
	if (!fp) {
		perror("fopen");
		exit(errno);
	}
	fp_test = fopen("file_test.h264" ,"wb+");
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