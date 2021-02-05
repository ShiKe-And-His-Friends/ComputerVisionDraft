extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavutil/pixdesc.h>
	#include <libavutil/hwcontext.h>
}

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <errno.h>

static int width, height;
static AVBufferRef* hw_device_ctx = nullptr;

using namespace std;

static void pgm_save(unsigned char* buf ,int wrap ,int xsize ,int ysize ,char* filename) {
	FILE* f;
	int i;
	fopen_s(&f ,filename, "w");
	fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
	for (i = 0; i < ysize; i++)
		fwrite(buf + i * wrap, 1, xsize, f);
	fclose(f);
}

static void decode(AVCodecContext* dec_ctx ,AVFrame* frame ,AVPacket* pkt ,const char* filename) {
	char buf[1024];
	int ret;
	ret = avcodec_send_packet(dec_ctx, pkt);
	if (ret < 0) {
		fprintf(stderr, "Error sending a packet for decoding\n");
		exit(1);
	}
	while (ret >= 0) {
		ret = avcodec_receive_frame(dec_ctx, frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			return;
		}else if (ret < 0) {
			fprintf(stderr, "Error during decoding\n");
			exit(1);
		}
		printf("saving frame %3d\n", dec_ctx->frame_number);
		fflush(stdout);
		snprintf(buf, sizeof(buf), "%s-%d", filename, dec_ctx->frame_number);
		pgm_save(frame->data[0], frame->linesize[0],
			frame->width, frame->height, buf);
	}
}

int main(int argc, char** argv) {

	int err;
	int width, height, size;
	const char* enc_name = "h264_vaapi";
	FILE* fin = nullptr, * fout = nullptr;
	AVCodec* codec;
	AVCodecContext* codecCtx;

	if (argc < 5) {
		fprintf(stderr, "Usage: %s <width> <height> <input file> <output file>", argv[0]);
		return -1;
	}
	width = atoi(argv[1]);
	height = atoi(argv[2]);
	size = width * height;
	fopen_s(&fin, argv[3], "r");
	if (!fin) {
		fprintf(stderr, "Fail to open input file \n");
		return -1;
	}

	fopen_s(&fout, argv[4], "w+b");
	if (!fout) {
		fprintf(stderr, "Fail to open output file \n");
		return -1;
	}
	err = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_VAAPI
		, nullptr, nullptr, 0);
	if (err < 0) {
		char info[256] = { 0 };
		av_strerror(err, info, sizeof(info));
		fprintf(stderr, "create hardware failure: %s \n", info);
		goto close;
	}
	codec = avcodec_find_decoder_by_name(enc_name);
	if (!codec) {
		err = -1;
		fprintf(stderr, "get hardware encoder failure \n");
		goto close;
	}
	codecCtx = avcodec_alloc_context3(codec);
	if (!codecCtx) {
		err = AVERROR(ENOMEM);
		fprintf(stderr, "get hardware encoder context failure \n");
		goto close;
	}


	codecCtx->width = width;
	codecCtx->height = height;
	codecCtx->time_base = AVRational {1, 25};
	codecCtx->framerate = AVRational{25, 1};
	codecCtx->sample_aspect_ratio = AVRational{1 ,1};
	codecCtx->pix_fmt = AV_PIX_FMT_VAAPI;
	if ((err = set_hwframe_ctx(codecCtx , hw_device_ctx)) < 0) {
	}

close:
	return 0;
}