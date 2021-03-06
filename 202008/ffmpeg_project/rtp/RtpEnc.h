#include <stdio.h>
#include "typedef.h"
#include "Network.h"

#ifndef RTP_ENC_H
#define RTP_ENC_H

#define H264_FRAME_RATE 25

#define RTP_PAYLOAD_MAX 1400

#define RTP_VERSION 2
#define RTP_H264 96
#define RTP_H264_SELF 107

typedef struct {
	uint8_t cache[RTP_PAYLOAD_MAX + 12]; //packet = RTP header + payload
	uint8_t buf[RTP_PAYLOAD_MAX]; //NAL: header + body
	uint8_t* buf_ptr;
	
	int aggregation; // 0 single 1 aggregation
	int payload_type; // 0 H.264/AVC 1 HEVC/H.265

	uint32_t ssrc;
	uint32_t seq;
	uint32_t timestamp;
}RtpContext;

int initRtpContext(RtpContext *context);

/* send a H.264/AVC video frame */
void  rtpSend(RtpContext *ctx ,UdpContext *udp ,const uint8_t *buf ,int size);

int finalizeRtpContext(RtpContext* context);

#endif
