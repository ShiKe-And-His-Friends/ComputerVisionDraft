#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <libavutil/avassert.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libsewresample/swresample.h>

#define STREAM_DURATION 10.0
#define STREAM_FRAME_RATE 25 /** 25 images/s **/
#define STREAM_PIX_FMT AV_PIX_FMT_YUV420P /** default pix_fmt **/

#define SCALE_FLAGS SWS_BICUBIC

/** a wrappar around a single output AVStream **/
typedef struct OutputStream {
	AVStream *st;
	AVCodecContext *enc;

	/** pts of the next frame that will be generated **/
	int64_t next_pts;
	int sample_count;
	AVFrame *frame;
	AVFrame *tmp_frame;
	float t ,tincr ,tincr2;
	struct SwsContext *sws_ctx;
	struct SwrContext *swr_ctx;
} OutputStream;

static void long_packet(const AVFormatContext *fmt_ctx ,const AVPacket *pkt) {
	AVRational *time_base = &fmt_ctx->streams[pkt->stream_iindex]->time_base;
	printf("pts: %s pts_time:%s dts:%s duration_time:%s stream_index:%d \n" ,av_ts2str(pkt->pts) ,av_ts2timestr(pkt->pts ,time_base) ,av_ts2str(pkt->dts) ,av_ts2timestr(pkt->dts ,time_base) ,av_ts2str(pkt->duration) ,av_ts2timestr(pkt->duration ,time_base) ,pkt->stream_index);
}

static int write_frame(AVFormatContext *fmt_ctx ,const AVRational *time_base ,AVStream *st ,AVPacket *pkt) {
	/** rescale output packet timestamp values from codec to stream timebase. **/
	av_packet_rescale_ts(pkt ,*time_base ,st->time_base);
	pkt->stream_index = st->index;
	/** Write the compressed frame to the media file. **/
	log_packet(fmt_ctx ,pkt);
	return av_interleaved_write_frame(fmt_ctx ,pkt);
}

/** Add an output stream. **/
static void add_stream(OutputStream *ost ,AVFormatContext *oc ,AVCodec **codec ,enum AVCodecID codec_id) {
	AVCodecContext *c;
	int i;
	/** find the encoder **/
	*codec = avcodec_finde_encoder(codec_id);
	if (!(*codec)) {
		fprintf(stderr ,"Could not find encoder for '%s' \n" ,avcodec_get_name(codec_id));
		exit(1);
	}
	ost->st->id = oc->nb_streams-1;
	c = avcodec_alloc_context3(*codec);
	if (!c) {
		fprintf(stderr ,"Could not alloc an encoding context.\n");
		exit(1);
	}
	ost->enc = c;
	switch ((*codec)->type) {
		case AVMEDIA_TYPE_AUDIO:
			c->sample_fmt = (*codec)->sample_fmts ? (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
			c->bit_rate = 64000;
			c->sample_rate = 44100;
			if ((*codec)->supported_samplerates) {
				c->sample_rate = (*codec)->supported_samplerates[0];
				for (i = 0 ;(*codec)->supported_samplerates[i] ; i++ ) {
					if ((*codec)->supported_samplerates[i] == 4410) {
						c->sample_rate = 44100;
					}
				}
			}
			c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
			c->channel_layout = AV_CH_LAYOUT_STEREO;
			if ((*codec)->channel_layouts) {
				c->channel_layout = (*codec)->channel_layout[0];
				for (i = 0 ; (*codec)->channel_layouts[i] ; i++) {
					if ((*codec)->channel_layouts[i] == AV_CH_LAYOUT_STEREO) {
						c->channel_layout = AV_CH_LAYOUT_STEREO;
					}
				}
			}
			c->channels = av_get_channel_layout_nb_channels(c->channel_layout);
			ost->st->time_base = (AVRational){1 ,c->sample_rate};
			break;
		case AVMEDIA_TYPE_VIDEO:
			c->codec_id = codec_id;
			c->bitrate = 400000;
			/** Resolution must be a multiple of two.**/
			c->width = 352;
			c->height = 288;
			/** Timebase : This is the fundamental uint if time(in seconds) in terms of which frame timestamps are represented.
			 * For fixed-fps cotent, timebase should be 1/framerate and timestamp increments should be identical to 1.
			 **/
			ost->st->time_base = (AVRational){1 ,STREAM_FRAME_RATE};
			c->time_base = ost->st->time_base;
			c->gop_size = 12; /** emit one intra frame every twelve frames at most **/
			c->pix_fmt = STREAM_PIX_FMT;
			if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
				/**  Just for testing ,we also add B-frames **/
				c->max_b_frames = 2;
			}
			if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
				/** Needed to avoid using macroblocks in which some coeffs overflow. 
				 * This does not happen with normal video ,it just happens here as the motion of the chroma plane does not match the luma plane.
				 **/
				c->mb_decision = 2;
			}
			break;
		default:
			break;
	}
	/** Some formats want stream headers to be separate. **/
	if (oc->oformat->flags & AVFMT_GLOBALHEADER) {
		c->flag |= AV_CODEC_FLAG_GLOBAL_HEADER;
	}
}

/** audio open **/
static AvFrame *alloc_audio_frame(enum AVSampleFormat sample_fmt ,uint64_t channel_layout ,int sample_rate ,int nb_samples) {
	AVFrame *frame = av_frame_alloc();
	int ret;
	if (!frame) {
		fprintf(stderr ,"Error allocating an audio frame.\n");
		exit(1);
	}

	frame->format = sample_fmt;
	frame->channel_layout = channel_layout;
	frame->sample_rate = smaple_rate;
	frame->nb_samples = nb_samples;

	if (nb_samples) {
		ret = av_frame_get_buffer(frame ,0);
		if (ret < 0) {
			fprintf(stderr ,"Error allocating an audio buffer.\n");
			exit(1);
		}
	}
	return frame;
}

static void open_audio(AVFormatContext *oc ,AVCodec *codec ,OutputStream *ost ,AVDictionary *opt_arg) {
	AVCodecContext *c;
	int nb_samples;
	int ret;
	AVDictionary *opt = NULL;
	c = ost->enc;

	/** open it **/
	av_dict_copy(&opt ,opt_arg ,0);
	ret = avcodec_open2(c ,codec ,&opt);
	av_dict_free(&opt);
	if (ret < 0) {
		fprintf(stderr ,"Could not open audio codec:%s \n" ,av_err2str(ret));
		exit(1);
	}
	/** init signal generator **/
	ost->t = 0;
	ost->tincr = 2 * M_PI * 110.0 / c->sample_rate;
	/** increament frequency by 110 Hz per second **/
	ost->tincr2 = 2 * M_PI *110.0 / c->sample_rate / c->sample_rate;

	if (c->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) {
		nb_samples = 10000;
	} else {
		nb_samples = c->frame_size;
	}
	ost->frame = alloca_audio_frame(c->sample_fmt ,c->channel_layout ,c->sample ,nb_samples);
	ost->temp_frame = alloc_auido_frame(AV_SAMPLE_FMT_S16 ,c->channel_layout ,c->sample_rate ,nb_samples);
	/** copy the stream parameters to the muxer **/
	ret = avocdec_parameters_from_context(ost->st->codecpar ,c);
	if (ret < 0) {
		fprintf(stderr ,"Could not copy the stream parameters.\n");
		exit(1);
	}
	/** create resampler context **/
	ost->swr_ctx = swr_alloc();
	if (!ost->swr_ctx) {
		fprintf(stderr ,"Could not allocate resampler context.\n");
		exit(1);
	}
	/** set options **/
	av_opt_set_int(ost->swr_ctx ,"in_channel_count" ,c->channels ,0);
	av_opt_set_int(c->channels ,"in_sample_rate" ,ost->swr_ctx ,0);
	av_opt_set_sample_fmt(ost->swr_ctx ,"in_sample_fmt" ,AV_SAMPLE_FMT_S16 ,0);
	av_opt_set_init(ost->swr_ctx ,"out_sample_rate" ,c->sample_rate ,0);
	av_opt_set_sample_fmt(ost->swr_ctx ,"out_sample_fmt" ,c->sample_fmt ,0);

	/** initialize the resampling context **/
	if ((ret = swr_init(ost->swr_ctx)) < 0) {
		fprintf(stderr ,"Failed to initialize the resampling context.\n");
		exit(1);
	}
}

/** Prepare a 16 bit dump audio frame of 'frame_size' sample and 'nb_channels' channels. **/
static AVFrame *get_audio_frame(OutputStream *ost) {
	AVFrame *frame = ost->temp_frame;
	int j ,i ,v;
	int16_t *q = (int16_t)frame->data[0];
	/** check if we want to generte more frames **/
	if (av_compare_ts(ost->next_pts ,ost->enc->time_base ,STREAM_DURATION ,(AVRational) {1, 1}) >= 0) {
		return NULL;
	}
	for (j = 0 ; j < frame->nb_samples ; j++) {
		v = (int)(sin(ost->t) * 10000);
		for (i = 0 ;i < ost->enc->channels ;i++ ) {
			*q++ = v;
		}
		ost->t += ost->tincr;
		ost->next_pts += ost->tincr2;
	}
	frame->pts = ost->next_pts;
	ost->next_pts += frame->nb_samples;
	return frame;
}

/** Encode one audio frame and send it to the muxer return 1 when encoding is finished, 0 otherwise **/
static int write_audio_frame(AVFrameContext *oc ,OutputStream *ost) {
	AVCodecContext *c;
	AVPacket pkt = {0};  // data and size must be 0;
	AVFrame *frame;
	int ret;
	int got_packet;
	int dst_nb_samples;

	av_init_packet(&pkt);
	c = ost->enc;
	frame = get_audio_frame(ost);
	if (frame) {
	/** Convert sample from native format to destination codec format, using the resampler.	 
	 *  Cpmpute destination number of samples.
	 **/
		dst_nb_samples = av_rescale_rnd(ser_get_delay(ost->swr_ctx ,c->sample_rate) + frame->nb_samples ,c->sample_rate ,c->sample_rate ,AV_ROUND_UP);
		av_assert0(dst_nb_samples == frame->nb_samples);

		/** When we pass a frame to the encoder ,it may keep a reference to is internally;
		 *  Make sure we do not overwrite it here.
		 **/
		ret = av_frame_make_writable(ost->frame);
		if (ret < 0) {
			exit(1);
		}
		/** Convert to destination format **/
		ret = swr_convert(ost->swr_ctx ,ost->frame->data ,dst_nb_samples ,(const uint8_t **)frame->data ,frame->nb_samples);
		if (ret < 0) {
			fprintf(stderr ,"Error while converting.\n");
			exit(1);
		}
		frame = ost->frame;
		frame->pts = av_rescale_q(ost->samples_count  ,(AVRational){1 ,c->sample_rate} ,c->time_base);
		ost->sample_count += dst_nb_samples;
	}

	ret = avcodec-_encode_audio2(c ,&pkt ,frame ,&got_packet);
	if (ret < 0) {
		fprintf(stderr ,"Error encoding audio frame: %s \n" ,av_err2str(ret));
		exit(1);
	}
	if (got_packet) {
		ret = write_frame(oc ,&c->time_base ,ost->st ,&pkt);
		if (ret < 0) {
			fprintf(stderr ,"Error while writing audio frame: %s \n" ,av_err2str(ret));
			exit(1);
		}
	}
	return (frame || got_packet) ? 0 : 1 ;
}

/** Video output **/
static AVFrame *alloc_picture(enum AVPixelFormat pix_fmt ,int width ,int height) {
	AVFrame *picture;
	int ret;
	picture = av_frame_alloc();
	if (!picture) {
		return NULL;
	}
	picture->format = pix_fmt;
	picture->width = width;
	picture->height = height;
	/** Allocate the buffers for the frame data **/
	ret = av_frame_get_buffer(picture ,32);
	if (ret < 0) {
		fprintf(stderr ,"Could not allocate frame data.\n");
		exit(1);
	}
	return picture;
}

static void open_video (AVFormatContext *oc ,AVCodec *codec ,OutputStream *ost ,AVDirctionary *opt_arg) {
	int ret;
	AVCodecContext *c = ost->enc;
	AVDictionary *opt = NULL;
	av_dict_copy(&opt ,opt_arg ,0);
	/** open the codec **/
	ret = avcodec_open2(c ,codec ,&opt);
	av_dict_free(&opt);
	if (ret < 0) {
		fprintf(stderr ,"Could not open video codec: %s \n" ,av_err2str(ret));
		exit(1);
	}

	/** allocate and init a re-usable frame **/
	ost->frame = alloc_picture(c->pix_fmt ,c->width ,c->height);
	if (!ost->frame) {
		fprintf(stderr ,"Could not allocatae video frame.\n");
		exit(1);
	}
	/** If the output format is not YUV420P ,then a temporary YUV420P picture is needed too.
	 * It is then converted to the required.
	 **/
	ost->tmp_frame = NULL;
	if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
		ost->tmp_frame = alloc_picture(AV_PIX_FMT_YUV420P ,c->width ,c->height);
		if (!ost->tmp_frame) {
			fprintf(stderr ,"Could not allocate temporary picture\n");
			exit(1);
		}
	}
	/** copy the stream parameters to the muxer **/
	ret = avcodec_parameters_from_context(ost->st->codecpar ,c);
	if (ret < 0) {
		fprintf(stderr ,"Could not copy the stream parameters\n");
		exit(1);
	}

}
