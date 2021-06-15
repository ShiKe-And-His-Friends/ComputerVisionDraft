#ifndef _H264_TO_RTO_H
#define _H264_TO_RTO_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "llist.h"

#define NAL_MAX 4000000
#define H264 96
#define H264_SELF 107
#define G711 8

typedef struct rtp_header {
	uint8_t csrc_len : 4;
	uint8_t extension : 1;
	uint8_t padding : 1;
	uint8_t version : 2;
	uint8_t payload_type : 7;
	uint8_t marker : 1;
	uint16_t seq_no;
	uint32_t timestamp;
	uint32_t ssrc;
} rtp_header_t;

typedef struct nalu_header {
	uint8_t type : 5;
	uint8_t nri : 2;
	uint8_t f : 1;
} nalu_header_t;

typedef struct nalu {
	int startcodeprefix_len;
	unsigned len;
	unsigned max_size;
	int forbidden_bit;
	int nal_reference_idc;
	int nal_uint_type;
	char* buf;
	unsigned short lost_packets;
} nalu_t;

typedef struct fun_indicator{
	uint8_t type : 5;
	uint8_t nri : 2;
	uint8_t f : 1;
} fu_indicator_t;

typedef struct fu_header {
	uint8_t type : 5;
	uint8_t r : 1;
	uint8_t e : 1;
	uint8_t s : 1;
} fu_header_t;

typedef struct rtp_package {
	rtp_header_t rtp_package_header;
	uint8_t* rtp_load;
} rtp_t;

struct func_para {
	uint8_t* send_buf;
	size_t len_sendbuf;
	linklist iplist;
};

int h264naltortp_send(int framerate ,uint8_t* pstStream ,int nalu_len ,void (*deal_func)(void* p) ,void *deal_func_para);

void add_client_to_list(linklist client_ip_list ,char* ipaddr);

#endif