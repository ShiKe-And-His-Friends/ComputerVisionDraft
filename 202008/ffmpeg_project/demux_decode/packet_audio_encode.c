#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <libavcodec/avcodec.h>

#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/frame.h>
#include <libavutil/samplefmt.h>

/** check that a given sample format is supported by the encoder **/
static int check_sample_fmt(const AVCodec *codec ,enum AVSampleFormat sample_fmt) {
	const enum AVSampleFormat *p = codec->sample_fmts;
	while (*p != AV_SAMPLE_FMT_NONE) {
		if (*p == sample_fmt) {
			return 1;
		}
		p++;
	}
	return 0;
}

/** just pick the hieghest supported samplerate **/
static int select_sample_rate(const AVCodec *codec) {
	const int *p;
	int best_samplerate = 0;
	if (!codec->supported_samplerates) {
		return 44100;
	}
	p = codec->supported_samplerates;
	while (*p) {
		if (!best_samplerate || abs(44100 - *p) < abs(44100 - best_samplerate)) {
			best_samplerate = *p;
		}
		p++;
	}
	return best_samplerate;
}

/** select layout with the hightest channel count **/
static int select_channel_layout(const AVCodec *codec) {
	const uint64_t *p;
	uint64_t best_ch_layout = 0;
	int best_nb_channels = 0;
	if (!codec->channel_layouts) {
		return AV_CH_LAYOUT_STEREO;
	}
	p = codec->channel_layouts;
	while (*p) {
		int nb_channels = av_get_channel_layout_nb_channels(*p);
		if (nb_channels > best_nb_channels) {
			best_ch_layout = *p;
			best_nb_channels = nb_channels;
		}
		p++;
	}
	return best_ch_layout;
}

static void encode(AVCodecContext *ctx ,AVFrame *frame ,AVPacket *pkt ,FILE *output) {
	int ret;
	/** send the frame for encoding **/
	ret = avcodec_send_frame(ctx ,frame);
	if (ret < 0) {
		fprintf(stderr ,"Error sending the frame to the encoder\n");
		exit(1);
	}
	/** read all the available output packet (in general there maty be any number of them) **/
	while (ret >= 0) {
		ret = avcodec_receive_packet(ctx ,pkt);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			return;
		} else if (ret < 0) {
			fprintf(stderr ,"Error encoding audio frame\n");
			exit(1);
		}
		fwrite(pkt->data ,1 ,pkt->size ,output);
		av_packet_unref(pkt);
	}
}

int main(int argc ,char **argv) {
	const char *filename;
	const AVCodec *codec;
	AVCodecContext *c = NULL;
	AVFrame *frame;
	AVPacket *pkt;
	int i ,j ,k ,ret;
	FILE *f;
	uint16_t *samples;
	float t,tincr;
	if (argc <= 1) {
		fprintf(stderr ,"Usage: %s <output file>\n"
				,argv[0]);
		return 0;
	}
	filename =argv[1];

	/** find the MP2 encoder **/
	codec = avcodec_find_encoder(AV_CODEC_ID_MP2);
	if (!codec) {
		fprintf(stderr ,"Could not found\n");
		exit(1);
	}

	c = avcodec_alloc_context3(codec);
	if (!c) {
		fprintf(stderr ,"Could not allocate audio codec context\n");
		exit(1);
	}
	/** put sample parameters **/
	c->bit_rate = 64000;
	/** check that the encoder supports s16 pcm input **/
	c->sample_fmt = AV_SAMPLE_FMT_S16;
	if (!check_sample_fmt(codec ,c->sample_fmt)) {
		fprintf(stderr ,"Encoder does not suppoert sample format %s"
				,av_get_sample_fmt_name(c->sample_fmt));
		exit(1);
	}
	/** select other audio parameters supported by the encoder **/
	c->sample_rate  = select_sample_rate(codec);
	c->channel_layout = select_channel_layout(codec);
	c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
	/** open it **/
	if (avcodec_open2(c ,codec ,NULL) < 0) {
		fprintf(stderr ,"Could not open codec\n");
		exit(1);
	}

	f = fopen(filename ,"wb");
	if (!f) {
		fprintf(stderr ,"could not allocate the packet\n");
		exit(1);
	}
	/** packet for holding encoded output **/
	pkt = av_packet_alloc();
	if (!pkt) {
		fprintf(stderr ,"could not allocate the packet\n");
		exit(1);
	}
	/** frame containing input raw audio **/
	frame = av_frame_alloc();
	if (!frame) {
		fprintf(stderr ,"Could not allocate audio frame\n");
		exit(1);
	}
	frame->nb_samples = c->frame_size;
	frame->format = c->sample_fmt;
	frame->channel_layout = c->channel_layout;

	/** allocate the data buffers **/
	ret = av_frame_get_buffer(frame ,0);
	if (ret < 0) {
		fprintf(stderr ,"Could not allocate audio data buffer\n");
		exit(1);
	}
	/** encode a single tone sound **/
	t = 0;
	tincr = 2 * M_PI * 440.0 / c->sample_rate;
	for (i = 0 ; i < 200 ; i++) {
		/** make sure the frame is writeable --make a copy if the encodr kept a reference inrternally **/
		ret = av_frame_make_writable(frame);
		if (ret < 0) {
			exit(1);
		}
		samples = (uint16_t*)frame->data[0];
		
		for (j = 0 ; j < c->frame_size ; j++) {
			samples[2*j] = (int)(sin(t) * 10000);
			for (k = 1 ; k < c->channels ; k++) {
				samples[2*j + k] = samples[2*j];
			}
			t += tincr;
		}
		encode(c ,frame ,pkt ,f);
	}
	/** flush the encoder **/
	encode(c ,NULL ,pkt ,f);
	fclose(f);
	av_frame_free(&frame);
	av_packet_free(&pkt);
	avcodec_free_context(&c);

	return 0;
}

/**
	1. struct
	1.1 
	
	2. function
	2.1 AVCodec *avcodec_find_encoder(enum AVCodecID id);
	2.2 (self define) static int check_sample_fmt(const AVCodec *codec ,enum AVSampleFormat sample_fmt);
	2.3 (self define) static int select_sample_rate(const AVCodec *codec);
	2.4 (self define) static int select_channel_layout(const AVCodec *codec);
	
	2.3 int av_frame_get_buffer(AVFrame *frame ,int algin);
	2.4 int av_frame_make_writeable(AVFrame *frame);
	2.5 (self define)static void encode(AVCodecContext *enc_ctx ,AVFrame *frame ,AVPacket *pkt ,FILE *outfile);
	2.6 int avcodec_receive_packet*(AVCodec *avctx ,AVPacket *avpkt);
	2.7 void av_packet_unref(AVPacket *pkt);
	
**/
