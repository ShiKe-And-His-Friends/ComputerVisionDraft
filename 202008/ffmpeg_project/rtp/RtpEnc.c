#include "RtpEnc.h"
#include "Utils.h"

int initRtpContext(RtpContext* context) {
	context->seq = 0;
	context->timestamp = 0;
	context->ssrc = 0x12345; // random number
	context->aggregation = 1;
	context->buf_ptr = context->buf;
	context->payload_type = 0; // 0 H.264/AVC  1 HECV/H.265
	return 0;
}

void rtpSend(RtpContext* ctx, UdpContext* udp, const uint8_t* buf, int size) {
	printf("rtp send start... \n");
	if (ctx == NULL || udp == NULL || buf == NULL || size <= 0) {
		printf("rtp data failure.\n");
		return;
	}

	uint8_t* r = buf;
	uint8_t* end = buf + size;

	uint8_t* r1;
	r1 = ff_avc_find_startcode(buf ,end);
	printf("\nindex = %p start = %p end = %p \n\n" ,r1 ,r ,end);

	uint32_t times = 0;

	while (r1 < end) {
		uint8_t* r2;
		while (!*(r1++)) {
			//skip current start codes
		}
		r2 = ff_avc_find_startcode(r1 ,end);
		//send a NALU rtp data
		printf("\n%d index1 = %p index2 = %p start = %p end = %p \n\n",times ++, r1 ,r2 ,r , end);

		r1 = r2;
	}

	printf("rtp send stop \n");
}