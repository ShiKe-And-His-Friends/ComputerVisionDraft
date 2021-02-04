extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavutil/frame.h>
	#include <libavutil/md5.h>
}

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

#define INBUF_SIZE 2048
#define AUDIO_REFILL_THRESH 4096

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

int main(int argc ,char** argv) {
	
	const char* outfilename, * filename;
	FILE *file;
	int ret ,len;
	uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
	uint8_t* data;
	size_t data_size;
	AVPacket* packet;
	AVCodec* codec;
	AVCodecContext* context;
	AVCodecParserContext* parser = nullptr;
	AVFrame* decoded_frame = nullptr;

	if (argc <= 2) {
		fprintf(stderr ,"Usage: %s <input_file> <ouput_file>\n" ,argv[0]);
		exit(2);
	}
	filename = argv[1];
	outfilename = argv[2];
	packet = av_packet_alloc();
	codec = avcodec_find_decoder(AV_CODEC_ID_MPEG1VIDEO);
	if (!codec) {
		fprintf(stderr ,"Parser not found.\n");
		return -1;
	}
	memset(inbuf + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);
	context = avcodec_alloc_context3(codec);
	if (!context) {
		fprintf(stderr, "context not found.\n");
		return -1;
	}
	parser = av_parser_init(codec->id);
	if (!context) {
		fprintf(stderr, "codec context not found.\n");
		return -1;
	}
	if (avcodec_open2(context ,codec ,nullptr) < 0) {
		fprintf(stderr, "open file failure.\n");
		return -1;
	}
	fopen_s(&file ,filename ,"rb");
	if (!file) {
		fprintf(stderr, "can not open file %s.\n" ,filename);
		return -1;
	}
	decoded_frame = av_frame_alloc();
	if (!decoded_frame) {
		fprintf(stderr, "can not alloc frame.\n");
		return -1;
	}
	while (!feof(file)) {
		data_size = fread(inbuf, 1, INBUF_SIZE, file);
		if (!data_size) {
			break;
		}
		data = inbuf;
		while (data_size > 0) {
			ret = av_parser_parse2(parser , context ,&packet->data ,&packet->size ,data ,data_size ,AV_NOPTS_VALUE ,AV_NOPTS_VALUE ,0);
			if (ret < 0) {
				fprintf(stderr, "Error while parser.\n");
				return -1;
			}
			data += ret;
			data_size -= ret;
			if (packet->size) {
				decode(context , decoded_frame ,packet , outfilename);
			}
		}
	}
	decode(context , decoded_frame ,packet , outfilename);
	fclose(file);
	avcodec_free_context(&context);
	av_parser_close(parser);
	av_frame_free(&decoded_frame);
	av_packet_free(&packet);
	return 0;
}