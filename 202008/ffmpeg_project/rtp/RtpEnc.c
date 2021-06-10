#include "RtpEnc.h"
#include "Utils.h"
#include "AVC.h"

static UdpContext* gUdpContext;

int initRtpContext(RtpContext* context) {
	context->seq = 0;
	context->timestamp = 0;
	context->ssrc = 0x12340000; // random number
	context->aggregation = 0;
	context->buf_ptr = context->buf;
	context->payload_type = 0; // 0 H.264/AVC  1 HECV/H.265
	return 0;
}

/*
*
*    0                   1                   2                   3
*    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
*   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   |V=2|P|X|  CC   |M|     PT      |       sequence number         |
*   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   |                           timestamp                           |
*   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*   |           synchronization source (SSRC) identifier            |
*   +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*   |            contributing source (CSRC) identifiers             |
*   :                             ....                              :
*   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*
**/
static void rtpSendData(RtpContext* ctx ,const uint8_t *buf ,int len ,int last) {
	int res = 0;
	// build rtp header
	uint8_t* pos = ctx->cache;
	pos[0] = (RTP_VERSION << 6) & 0xff; // V P X CC
	pos[1] = (uint8_t)((RTP_H264 & 0x07f) | ((last & 0x01) << 7)); // M PayloadType
	Load16(&pos[2] ,(uint16_t)ctx->seq);
	Load32(&pos[4] ,ctx->timestamp);
	Load32(&pos[8] ,ctx->ssrc);

	//copy audio/video data
	memcpy(&pos[12] ,buf ,len);

	// send socket udp stream
	if (ctx != NULL && gUdpContext != NULL) {
		res = udpSend(gUdpContext ,ctx->cache ,(uint32_t)(len + 12));	
	} else {
		printf("rtp socket error.\n");
	}
	
	// debug print	
	printf("\nrtp sned data cache [%d]:" ,res);
	for (int i = 0; i < 20; i++) {
		printf("%.2X", ctx->cache[i]);
	}
	printf("\n");

	// clear buffer
	memset(ctx->cache ,0 ,RTP_PAYLOAD_MAX + 10);

	ctx->buf_ptr = ctx->buf;
	ctx->seq = (ctx->seq + 1) & 0xffff;
}

static void rtpSenNAL(RtpContext* ctx, const uint8_t* nal, int size, int last) {
	//printf("rtp send nal len = %d\n" ,size);
	if (size <= RTP_PAYLOAD_MAX) {
		// Aggregation Packets
		if (ctx->aggregation) {
			int buffered_size = (int)(ctx->buf_ptr - ctx->buf);
			uint8_t curNRI = (uint8_t)(nal[0]& 0x60); // NRI
			
			// aggregate other nal data
			if (buffered_size + 2 + size > RTP_PAYLOAD_MAX) {
				rtpSendData(ctx ,ctx->buf ,buffered_size ,0);
				buffered_size = 0;
			}
			
			// set nal's emergency permissions
			if (buffered_size == 0) {
				*ctx->buf_ptr++ = (uint8_t)(24 | curNRI);
			}
			else {
				uint8_t lastNRI = (uint8_t)(ctx->buf[0] & 0x60);
				if (curNRI > lastNRI) {
					// use new NRI
					ctx->buf[0] = (uint8_t)((ctx->buf[0] & 0x9f) | curNRI);
				
				}
			}
			
			// set Final bit 
			ctx->buf[0] |= (nal[0] & 0x80);

			//copy nal data
			Load16(ctx->buf_ptr ,(uint16_t)size); // set new NAL size
			ctx->buf_ptr += 2;
			memcpy(ctx->buf_ptr ,nal ,size); // set new NAL header & data
			ctx->buf_ptr += size;

			if (last == 1) {
				rtpSendData(ctx ,ctx->buf ,(int)(ctx->buf_ptr - ctx->buf) ,1);
			}
		}
		else {
			rtpSendData(ctx ,nal ,size ,last);
		}
	}
	else {
		// cut small slice
		if (ctx->buf_ptr > ctx->buf) {
			send(ctx ,ctx->buf ,(int)(ctx->buf_ptr - ctx->buf) ,0);
		}
		int headerSize;
		uint8_t* buf = ctx->buf;
		uint8_t type = nal[0] & 0x1f;
		uint8_t nri = nal[0] & 0x60;

		// set NAL slice A
		buf[0] = 28; // slice type FU-A
		buf[0] |= nri;

		buf[1] = type;
		buf[1] |= 1 << 7;
		headerSize = 2;
		size -= 1;
		nal += 1;

		while (size + headerSize > RTP_PAYLOAD_MAX) {
			memcpy(&buf[headerSize] ,nal ,(size_t)(RTP_PAYLOAD_MAX - headerSize));
			rtpSendData(ctx ,buf ,RTP_PAYLOAD_MAX ,0);
			nal += RTP_PAYLOAD_MAX - headerSize;
			size -= RTP_PAYLOAD_MAX - headerSize;
			buf[1] &= 0x7f; //set slice type S(tart) flag = 0
		}
		buf[1] |= 0x40; //set slice type E(nd) flag = 0
		memcpy(&buf[headerSize] ,nal ,size);
		rtpSendData(ctx ,buf ,size + headerSize ,last);
	}
}

void rtpSend(RtpContext* ctx, UdpContext* udp, const uint8_t* buf, int size) {
	printf("rtp send start... \n");
	if (ctx == NULL || udp == NULL || buf == NULL || size <= 0) {
		printf("rtp data failure.\n");
		return;
	}

	const uint8_t* r = buf;
	const uint8_t* end = buf + size;
	gUdpContext = udp;

	r = ff_avc_find_startcode(buf ,end);
	printf("\nindex = %p start = %p end = %p \n\n" ,r ,r ,end);

	uint32_t times = 0;

	while (r < end) {
		const uint8_t* r1;
		while (!*(r++)) {
			//skip current start codes
		};
		r1 = ff_avc_find_startcode(r ,end);
		
		//send a NALU rtp data
		
		//printf("\n%d index1 = %p index2 = %p start = %p end = %p  length = %d \n\n",times ++, r ,r1  ,r , end ,(r1 - r));
		rtpSenNAL(ctx ,r ,(int)(r1 - r) ,r1 == end);

		Sleep(1000 / H264_FRAME_RATE);
		ctx->timestamp += (90000.0 / H264_FRAME_RATE);
		r = r1;
	}

	printf("rtp send stop \n");
}