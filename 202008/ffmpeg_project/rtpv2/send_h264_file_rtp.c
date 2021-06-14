/**
* Thanks for https://github.com/hmgle/h264_to_rtp
* Copy from it.
*/

#include <stdio.h>
#include <stdlib.h>
#include "send_h264_file_rtp.h"

void add_client(linklist client_ip_list ,char*  ipaddr) {
	struct sockaddr_in server_c;
	pnode pnode_tmp;
	const int on = 1;

	insert_nodelp_node(client_ip_list ,ipaddr);
	pnode_temp = search_node(CLIENT_IP_LIST ,ipaddr);
	server_c.sin_family = AF_INET;
	server_c.sin_port = htons(DEST_PORT);

	if ((connect(pnode_temp->node_info.socket_c ,(const struct sockaddr *)&server_c)) == -1) {
		perror("connect");
		exit(-1);
	}
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
		ret = fread(tmpbuf2, 1, 1, fp);
		if (!ret) {
			return -1;
		}
	
	} while (1);

}