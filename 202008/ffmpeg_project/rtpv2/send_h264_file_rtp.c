/**
* Thanks for https://github.com/hmgle/h264_to_rtp
* Copy from it.
*/

#include <stdio.h>
#include <stdlib.h>
#include "send_h264_file_rtp.h"

#define RTP_PAYLOAD_MAX_SIZE 1400
#define SSRC_NUM 44
#define SEND_BUF_SIZE 1500
#define DEFAULT_DEST_PORT 1234
#define NAL_BUF_SIZE 1500 * 150

static int max;
uint16_t DEST_PORT;
linklist CLIENT_IP_LIST;
uint8_t SENDBUFFER[SEND_BUF_SIZE];
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

static void send_data_to_client_list(uint8_t* send_buf, size_t len_sendbuf, linklist client_ip_list) {

}

static int h264nal2rtp_send(int frameRate ,uint8_t* pstStream ,int nalu_len ,linklist client_ip_list) {
	uint8_t* nalu_buf;
	nalu_buf = pstStream;
	static uint32_t ts_current = 0;
	static uint16_t seq_num = 0;
	rtp_header_t* rtp_hdr;
	nalu_header_t* nalu_hdr;
	fu_indicator_t* fu_ind;
	fu_header_t* fu_hdr;
	size_t len_sendbuf;

	int fu_pack_num; //nal num
	int last_fu_pack_size; //last slice size
	int fu_seq; //fu-a 

	ts_current += (90000 / frameRate);

	if (nalu_len < 1) {
		return -1;
	}

	if (nalu_len <= RTP_PAYLOAD_MAX_SIZE) {
		memset(SENDBUFFER ,0 ,SEND_BUF_SIZE);
		rtp_hdr = (rtp_header_t*)SENDBUFFER;
		rtp_hdr->csrc_len = 0;
		rtp_hdr->extension = 0;
		rtp_hdr->padding = 0;
		rtp_hdr->version = 2;
		rtp_hdr->payload_type = H264;
		rtp_hdr->seq_no = htons(++seq_num % UINT16_MAX);
		rtp_hdr->timestamp = htonl(ts_current);
		rtp_hdr->ssrc = htonl(SSRC_NUM);
		
		nalu_hdr = (nalu_header_t*)&SENDBUFFER[12];
		nalu_hdr->f = (nalu_buf[0] & 0x80) >> 7;
		nalu_hdr->nri = (nalu_buf[0] & 0x60) >> 5;
		nalu_hdr->type = (nalu_buf[0] & 0x1f);

		memcpy(SENDBUFFER + 13 ,nalu_buf + 1 ,nalu_len - 1);

		len_sendbuf = 12 + nalu_len;
		send_data_to_client_list(SENDBUFFER ,len_sendbuf ,client_ip_list);
	}
	else {
		// cut fu-a

		fu_pack_num = nalu_len % RTP_PAYLOAD_MAX_SIZE ? (nalu_len / RTP_PAYLOAD_MAX_SIZE + 1) : nalu_len / RTP_PAYLOAD_MAX_SIZE;
		last_fu_pack_size = nalu_len % RTP_PAYLOAD_MAX_SIZE ? nalu_len % RTP_PAYLOAD_MAX_SIZE : RTP_PAYLOAD_MAX_SIZE;
		fu_seq = 0;
		for (fu_seq = 0; fu_seq < fu_pack_num;fu_seq ++) {
			memset(SENDBUFFER ,0 ,SEND_BUF_SIZE);
			if (fu_seq == 0) {
				rtp_hdr = (rtp_header_t*)SENDBUFFER;
				rtp_hdr->csrc_len = 0;
				rtp_hdr->extension = 0;
				rtp_hdr->padding = 0;
				rtp_hdr->version = 2;
				rtp_hdr->payload_type = H264;
				rtp_hdr->marker = 0;
				rtp_hdr->seq_no = htons(++seq_num % UINT16_MAX);
				rtp_hdr->timestamp = htonl(ts_current);
				rtp_hdr->ssrc = htonl(SSRC_NUM);

				fu_ind = (fu_indicator_t*)&SENDBUFFER[12];
				fu_ind->f = (nalu_buf[0] & 0x80) >> 7;
				fu_ind->nri = (nalu_buf[0] & 0x60) >> 5;
				fu_ind->type = 28;

				fu_hdr = (fu_header_t*)&SENDBUFFER[13];
				fu_hdr->s = 1;
				fu_hdr->e = 0;
				fu_hdr->r = 0;
				fu_hdr->type = nalu_buf[0] & 0x1f;

				memcpy(SENDBUFFER + 14 ,nalu_buf + 1 ,RTP_PAYLOAD_MAX_SIZE - 1);

				len_sendbuf = 12 + 2 + (RTP_PAYLOAD_MAX_SIZE - 1);
				send_data_to_client_list(SENDBUFFER ,len_sendbuf ,client_ip_list);
			}
			else if (fu_seq < fu_pack_num - 1){
				rtp_hdr = (rtp_header_t*)SENDBUFFER;
				rtp_hdr->csrc_len = 0;
				rtp_hdr->extension = 0;
				rtp_hdr->padding = 0;
				rtp_hdr->version = 2;
				rtp_hdr->payload_type = H264;
				rtp_hdr->marker = 0;
				rtp_hdr->seq_no = htons(++seq_num % UINT16_MAX);
				rtp_hdr->timestamp = htonl(ts_current);
				rtp_hdr->ssrc = htonl(SSRC_NUM);

				fu_ind = (fu_indicator_t*)&SENDBUFFER[12];
				fu_ind->f = (nal_buf[0] & 0x80) >> 7;
				fu_ind->nri = (nal_buf[0] & 0x60) >> 5;
				fu_ind->type = 28;

				fu_hdr = (fu_header_t*)&SENDBUFFER[13];
				fu_hdr->s = 0;
				fu_hdr->e = 0;
				fu_hdr->r = 0;
				fu_hdr->type = nalu_buf[0] & 0x1f;

				memcpy(SENDBUFFER + 14 ,nalu_buf + RTP_PAYLOAD_MAX_SIZE * fu_seq ,RTP_PAYLOAD_MAX_SIZE);
				len_sendbuf = 12 + 2 + RTP_PAYLOAD_MAX_SIZE;
			}
			else {
				rtp_hdr = (rtp_header_t*)SENDBUFFER;
				rtp_hdr->csrc_len = 0;
				rtp_hdr->extension = 0;
				rtp_hdr->padding = 0;
				rtp_hdr->version = 2;
				rtp_hdr->payload_type = H264;
				rtp_hdr->marker = 1;
				rtp_hdr->seq_no = htons(++seq_num % UINT16_MAX);
				rtp_hdr->timestamp = htonl(ts_current);
				rtp_hdr->ssrc = htonl(SSRC_NUM);

				fu_ind = (fu_indicator_t*)&SENDBUFFER[12];
				fu_ind->f = (nalu_buf[0] & 0x80) >> 7;
				fu_ind->nri = (nalu_buf[0] & 0x60) >> 5;
				fu_ind->type = 28;

				fu_hdr = (fu_header_t*)&SENDBUFFER[13];
				fu_hdr->s = 0;
				fu_hdr->e = 1;
				fu_hdr->r = 0;
				fu_hdr->type = nalu_buf[0] & 0x1f;

				memcpy(SENDBUFFER + 14 ,nalu_buf + RTP_PAYLOAD_MAX_SIZE * fu_seq ,last_fu_pack_size);
				len_sendbuf = 12 + 2 + last_fu_pack_size;
				send_data_to_client_list(SENDBUFFER ,len_sendbuf ,client_ip_list);
			}
		}
	}
	return 0;
}

void add_client(linklist client_ip_list ,char*  ipaddr) {
	
}

static int copy_nal_from_file(FILE* fp, uint8_t* buf, int* len) {
	char tmpbuf[4];
	char tmpbuf2[1];
	int flag = 0;
	int ret;

	*len = 0;
	do {
		ret = fread(tmpbuf2, 1, 1, fp);
		if (!ret) {
			return -1;
		}
		if (!flag && tmpbuf2[0] != 0x00) {
			//printf("buf index %d\n", *len);
			if (*len > max) {
				max = *len;
				//printf("max index %d\n", *len);
			}
			buf[*len] = tmpbuf2[0];
			(*len)++;
		}
		else if (!flag && tmpbuf2[0] == 0x00) {
			flag = 1;
			tmpbuf[0] = tmpbuf2[0];
		}
		else if (flag) {
			switch (flag) {
				case 1:
					if (tmpbuf2[0] == 0x00) {
						flag++;
						tmpbuf[1] = tmpbuf2[0];
					}
					else {
						flag = 0;
						buf[*len] = tmpbuf[0];
						(*len)++;
						buf[*len] = tmpbuf2[0];
						(*len)++;
					}
					break;

				case 2:
					if (tmpbuf2[0] == 0x0) {
						flag++;
						tmpbuf[2] = tmpbuf2[0];
					}
					else if (tmpbuf2[0] == 0x1) {
						flag = 0;
						return *len;
					}
					else {
						flag = 0;
						buf[*len] = tmpbuf[0];
						(*len)++;
						buf[*len] = tmpbuf[1];
						(*len)++;
						buf[*len] = tmpbuf2[0];
						(*len)++;
					}
					break;

				case 3:
					if (tmpbuf2[0] == 0x1) {
						flag = 0;
						//printf("len is %d\n" ,*len);
						return *len;
					}
					else {
						flag = 0;
						break;
					}
			}
		}
	} while (1);
	return *len;
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