extern "C" {
	#include <libavcodec/avcodec.h>
	#include <libavutil/frame.h>
	#include <libavutil/md5.h>
}
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

#define AUDIO_INBUF_SIZE 20480
#define AUDIO_REFILL_THRESH 4096

using namespace std;

static void decode(AVCodecContext* dec_ctx ,AVPacket* pkt ,AVFrame* frame ,FILE* outfile) {
	int i, ch;
	int ret, data_size;
	ret = avcodec_send_packet(dec_ctx ,pkt);
	if (ret < 0) {
		fprintf(stderr, "Error submitting the packet to the decoder\n");
		exit(1);
	}
	while (ret >= 0) {
		ret = avcodec_receive_frame(dec_ctx ,frame);
		if (ret = AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			return;
		}
		else if (ret < 0) {
			fprintf(stderr, "Error during decode\n");
			exit(1);
		}
		if (data_size < 0) {
			fprintf(stderr, "Failed to calculate data size\n");
			exit(1);
		}
		for (i = 0; i < frame->nb_samples; i++) {
			for (ch = 0; ch < dec_ctx->channels; ch++) {
				fwrite(frame->data[ch] + data_size * i, 1, data_size, outfile);
			}
		}
	}
}

int main(int argc ,char** argv) {
	
	const char* outfilename, * filename;
	FILE *file , *outfile;
	int ret ,len;
	uint8_t inbuf[AUDIO_INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
	uint8_t* data;
	size_t data_size;
	AVPacket* packet;
	AVCodec* codec;
	AVCodecContext* context;
	AVCodecParserContext* parser = nullptr;
	AVFrame* decoded_frame = nullptr;

	if (argc <= 2) {
		fprintf(stderr ,"Usage: %s <input_file> <ouput_file>\n" ,argv[0]);
		exit(0);
	}
	filename = argv[1];
	outfilename = argv[2];
	packet = av_packet_alloc();
	codec = avcodec_find_decoder(AV_CODEC_ID_MP3);
	if (!codec) {
		fprintf(stderr ,"Parser not found.\n");
		return -1;
	}
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
	file = fopen(filename ,"rb");
	if (!file) {
		fprintf(stderr, "can not open file %s.\n" ,filename);
		return -1;
	}
	outfile = fopen(outfilename, "wb");
	if (!outfile) {
		fprintf(stderr, "can not open output file %s.\n", outfilename);
		return -1;
	}
	data = inbuf;
	data_size = fread(inbuf ,1 , AUDIO_INBUF_SIZE ,file);
	while (data_size > 0) {
		if (!decoded_frame) {
			if (!(decoded_frame = av_frame_alloc())) {
				fprintf(stderr, "can not alloc decode audio frame.\n");
				return -1;
			}
		}
		ret = av_parser_parse2(parser ,context ,&packet->data ,&packet->size ,data ,data_size ,AV_NOPTS_VALUE ,AV_NOPTS_VALUE ,0);
		if (ret < 0) {
			fprintf(stderr, "parser parser2 failure.\n");
			return -1;
		}
		data += ret;
		data_size -= ret;
		if (packet->size) {
			decode(context ,packet ,decoded_frame ,outfile);
		}
		if (data_size < AUDIO_REFILL_THRESH) {
			memmove(inbuf ,data ,data_size);
			data = inbuf;
			len = fread(data + data_size ,1 ,AUDIO_INBUF_SIZE - data_size ,f);
			if (len > 0) {
				data_size += len;
			}
		}
	}
	packet->data = nullptr;
	packet->size = 0;
	decode(context ,packet ,decoded_frame ,outfile);
	fclose(file);
	fclose(outfile);
	avcodec_free_context(&context);
	av_parser_close(parser);
	av_frame_free(&decoded_frame);
	av_packet_free(&packet);
	return 0;
}