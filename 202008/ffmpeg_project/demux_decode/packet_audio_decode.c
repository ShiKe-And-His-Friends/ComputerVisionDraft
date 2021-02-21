#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/frame.h>
#include <libavutil/mem.h>
#include <libavcodec/avcodec.h>

#define AUDIO_INBF_SIZE 20480
#define AUDIO_REFILL_THRESH 4096

static void decode(AVCodecContext *dec_ctx ,AVPacket *pkt ,AVFrame *frame ,FILE *outfile) {
	int i ,ch;
	int ret ,data_size;
	/** send packet with compressed data to decoder **/
	ret = avcodec_send_packet(dec_ctx ,pkt);
	if (ret < 0) {
		fprintf(stderr ,"Error submitting the packet to the decoder. \n \n");
		exit(1);
	}

	/** read all output frames(in general there may be any nember of them) **/
	while (ret >= 0) {
		ret = avcodec_receive_frame(dec_ctx ,frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			return;
		} else if (ret < 0) {
			fprintf(stderr ,"Error during doceding.\n");
			exit(1);
		}
		data_size = av_get_bytes_per_sample(dec_ctx->sample_fmt);
		if (data_size < 0) {
			/** This should not occur ,check just for paranoia **/
			fprintf(stderr ,"Failed to calculate data size.\n");
			exit(1);
		}
		for (i = 0 ;i < frame->nb_samples ;i++) {
			for (ch = 0 ;ch < dec_ctx->channels ;ch++) {
				fwrite(frame->data[ch] + data_size * i ,1 ,data_size ,outfile);
			}
		}
	}
}

int main(int argc ,char **argv){

	const char *outfilename ,*filename;
	const AVCodec *codec;
	AVCodecContext *c = NULL;
	AVCodecParserContext *parser = NULL;
	int len,ret;
	FILE *f, *outfile;
	uint8_t inbuf[AUDIO_INBF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
	uint8_t *data;
	size_t data_size;
	AVPacket *pkt;
	AVFrame *decoded_frame = NULL;

	if (argc <= 2) {
		fprintf(stderr ,"Usage:%s <input file> <output file> \n" ,argv[0]);
		exit(0);
	}
	filename = argv[1];
	outfilename = argv[2];
	pkt = av_packet_alloc();

	/** find MPEG audio decoder **/
	codec = avcodec_find_decoder(AV_CODEC_ID_MP3);
	if (!codec) {
		fprintf(stderr ,"Codec not find.\n");
		exit(1);
	}
	parser = av_parser_init(codec->id);
	if (!parser) {
		fprintf(stderr ,"Parser not found.\n");
		exit(1);
	}
	c = avcodec_alloc_context3(codec);
	if (!c){
		fprintf(stderr ,"Could not allocate codec context.\n");
		exit(1);
	}
	/** open it **/
	if (avcodec_open2(c ,codec ,NULL) < 0){
		fprintf(stderr ,"Could not open codec.\n");
		exit(1);
	}

	f = fopen(filename ,"rb");
	if(!f){
		fprintf(stderr ,"Could not open %s \n" ,filename);
		exit(1);
	}
	outfile = fopen(outfilename ,"wb");
	if (!outfile) {
		av_free(c);
		exit(1);
	}

	/** decode until eof **/

	data = inbuf;
	data_size = fread(inbuf ,1 ,AUDIO_INBF_SIZE ,f);
	while (data_size > 0) {
		if (!decoded_frame) {
			if (!(decoded_frame = av_frame_alloc())) {
				fprintf(stderr ,"Could not allocate frame.\n");
				exit(1);
			}
		}
		ret = av_parser_parse2(parser ,c ,&pkt->data ,&pkt->size
				,data ,data_size ,AV_NOPTS_VALUE ,AV_NOPTS_VALUE ,0);
		if (ret < 0) {
			fprintf(stderr ,"Error while pasring\n");
			exit(1);
		}
		data += ret;
		data_size -= ret;

		if (pkt->size) {
			decode(c ,pkt ,decoded_frame ,outfile);
		}
		if (data_size < AUDIO_REFILL_THRESH) {
			memmove(inbuf ,data ,data_size);
			data = inbuf;
			len = fread(data + data_size ,1 ,AUDIO_INBF_SIZE - data_size ,f);
			if (len > 0) {
				data_size += len;
			}
		}
	}

	/** flush decoder **/
	pkt->data = NULL;
	pkt->size = 0;
	decode(c ,pkt ,decoded_frame ,outfile);

	fclose(outfile);
	fclose(f);

	avcodec_free_context(&c);
	av_parser_close(parser);
	av_frame_free(&decoded_frame);
	av_packet_free(&pkt);
	
	return 0;
}

/**
	1. struct
	1.1 typedef strut AVCodec { } AVCodec;
	1.2 typedef struct AVCodecContext {} AVCodecContext;
	1.3 typedef struct AVCodecParserContext {} AVCodecContext;
	1.4 typedef struct AVPacket {} AVPacket;
	1.5 typedef struct AVFrame {} AVFrame;
	1.6 enum AVCodecID {};
	1.7* enum AVSampleFormat {};
	
	2. function
	2.1 AVPacket *av_packet_alloc(void);
	2.2 AVCodec *avcodec_find_decoder(enum AVCodecID id);
	2.3 AVCodecParser *av_parser_init(int codec_id);
	2.4 AVCodecContext *avcodec_alloc_context3(const AVCodec *codec);
	2.5 int avcodec_open2(AVCodecContext *avctx ,const AVCodec *codec ,AVDictionary **options);
	2.6 AVFrame *av_frame_alloc(void);
	2.7 int av_parser_parser2(AVCodecParserContext *s ,AVCodecContext *avctx 
				,uint8_t **pountbuf ,int *poutbuf_size
				,const uint8_t *buf ,int buf_size
				,int64_t pts ,int64_t dts
				,int64_t pos);
	2.8 (selfdefine) static void decode(AVCodecContext avc_ctx ,AVPacket pkt ,AVFrame frame ,FILE *outfile);
	2.9 int avcodec_send_packet(AVCodecContext *avctx ,const AVPacket *avpacket);
	2.10 int avcodec_receive_frame(AVCodecContext *avctx ,AVFrame *frame);
	2.11 int av_get_byte_per_sample(enum AVSampleFormat sample_fmt);
	
**/
