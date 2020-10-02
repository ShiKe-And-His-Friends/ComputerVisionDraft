#include <stdio.h>
#include <errno.h>
#include <libavutil/hwcontext.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

static AVFormatContext *ifmt_ctx = NULL ,*ofmt_ctx = NULL;
static AVBufferRef *hw_device_ctx = NULL;
static AVCodecContext *decoder_ctx = NULL ,*encoder_ctx = NULL;
static int video_stream = -1;
static AVStream *ost;
static int initialized = 0;

static enum AVPixelFormat get_vaapi_format(AVCodecContext *ctx ,const enum AVPixelFormat *pix_fmts) {
	const enum AVPixelFormat *p;
	for (p = oix_fmts ; *p != AV_PIX_FMT_NONE ;p ++) {
		if (*p == AV_PIX_FMT_VAAPI) {
			return *p;
		}
	}
	fprintf(stderr ,"Unable to decode this file using VA_API.\n");
	return AV_PIX_FMT_NONE;
}

static int open_input_file (const char *filename) {
	int ret;
	AVCodec *decoder = NULL:
	AVStream *video = NULL;
	if ((ret = avformat_open_input(&ifmt_ctx ,filename ,NULL ,NULL)) < 0) {
		fprintf(stderr ,"Cannot open input file '%s' ,Error code:%s \n" ,filename ,av_err2str(ret));
		return ret;
	}
	if ((ret = avformat_find_stream_info(ifmt_ctx ,NULL)) < 0) {
		fprintf(stderr ,"Cannot find input stream information. Error code:%s\n" ,av_err2str(ret));
		return ret;
	}
	ret = av_find_best_stream(ifmt_ctx ,AVMEDIA_TYPE_VIDEO ,-1 ,-1 ,&decoder ,0);
	if (ret < 0) {
		fprintf(stderr ,"Cannot find a video stream in the input file. Error code:%s\n" ,av_err2str(ret));
		return ret;
	}
	video_stream = ret;
	if (!(decoder_ctx = avcodec_alloc_context3(decoder))) {
		return AVERROR(ENOMEM);
	}
	video = ifmt_ctx->streams[video_stream];
	if ((ret = avcodec_parameters_to_context(decoder_ctx ,video->codecpar)) < 0) {
		fprintf(stderr ,"avcodec_parameters_to_context error. Error code:%s \n" ,av_err2str(ret));
		return ret;
	}
	decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
	if (!decoder_ctx->hw_device_ctx) {
		fprintf(stderr ,"A hardware device reference create failed.\n");
		return AVERROR(ENOMEM);
	}
	decoder_ctx->get_frame = get_vaapi_format;
	if ((ret = avcodec_open2(decoder_ctx ,decoder ,NULL)) < 0) {
		fprintf(stderr ,"Failed to open codec for decoding. Error code:%s \n" ,av_err2str(ret));
	}
	return ret;
}

static int encode_write(AVFrame *frame) {
	int ret = 0;
	AVPacket enc_pkt;
	av_init_packet(&enc_pkt);
	enc_pkt.data = NULL;
	enc_pkt.size = 0;
	if ((ret = avcodec_send_frame(encoder_ctx ,frame)) < 0) {
		fprintf(stderr,"Error during encodeig.Error code:%s \n" ,av_err2str(ret));
		goto end;
	}
	while (1) {
		ret = avcodec_receive_packet(ecoder_ctx ,&enc_pkt);
		if (ret) {
			break;
		}
		enc_pkt.stream_index = 0;
		av_packet_rescale_ts(&enc_pkt ,ifmt_ctx->streams[video_stream]->time_base ,ofmt_ctx->streams[0]->time_base);
		ret = av_interleaved_write_frame(ofmt_ctx ,&enc_pkt);
		if (ret < 0) {
			fprintf(stderr ,"Error during writing data to output file. Error code:%s\n" ,av_err2str(ret));
			return -1;
		}
	}

end:
	if (ret == AVERROR_EOF) {
		return 0;
	}
	ret = ((ret == AVERROR(EAGAIN)) ? 0 :-1);
	return ret;
}

static int dec_enc(AVPacket *pkt ,AVCodec *enc_codec) {
	AVFrame *frame;
	int ret = 0;
	ret = avcodec_send_packet(decoder_ctx ,pkt);
	if (ret < 0) {
		fprintf(stderr ,"Error during decoding.Error code:%s\n" ,av_err2str(ret));
		return ret;
	}
	while (ret >= 0) {
		if (!(frame = av_frame_alloc())) {
			return AVERROR(ENOMEM);
		}
		ret = avcodec_receive_frame(decoder_ctx ,frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			av_frame_free(&frame);
			return 0;
		} else if (ret < 0) {
			fprintf(stderr ,"Error while decoding.Error code:%s \n" ,av_err2str(ret));
			goto fail;
		}
		if (!initialized) {
			/** we need to ref hw_frames_ctx of decoder to initialize encoder's codec.
			 * Only after we get a decoded frame ,can we obtain its hw_frames_ctx
			 **/
			encoder_ctx->hw_frames_ctx = av_buffer_ref(decoder_ctx->hw_frames_ctx);
			if (!encoder_ctx->hw_frames_ctx) {
				ret = AVERROR(ENOMEM);
				goto fail;
			}
			/**
			 * set AVCodecContext parameters for encoder,here we keep them stay.
			 * the same as decoder.
			 * xxx:now the sample can't handle resolution change case.
			 **/
			encoder_ctx->time_base = av_inv_q(decoder_ctx->framerate);
			encoder_ctx->pix_fmt = AV_PIX_FMT_VAAPI;
			encoder_ctx->width = decoder_ctx->width;
			encoder_ctx->height = decoder_ctx->height;
			if ((ret = avcodec_open2(encoder_ctx ,enc_codec ,NULL)) < 0) {
				fprintf(stderr,"Failed to open encode codec. Error code:%s\n" ,av_err2str(ret));
				goto fail;
			}
			if (!(ost = avformat_new_stream(ofmt_ctx ,enc_codec))) {
				fprintf(stderr ,"Failed to allocate stream for output format.\n");
				ret = AVERROR(ENOMEM);
				goto fail;
			}
			ost->time_base = encoder_ctx->time_base;
			ret = avcodec_parameters_from_context(ost->codecpar ,encodr_ctx);
			if (ret < 0) {
				fprintf(stderr ,"Failed to copy the stream parameters. Error code;%s\n" ,av_err2str(ret));
				goto fail;
			}
			/** write the stream header **/
			if ((ret = avformat_write_header(ofmt_ctx ,NULL)) < 0) {
				fprintf(stderr,"Error while writing stream header. Error code:%s \n" ,av_err2str(ret));
				goto fail;
			}
			initialize = 1;
		}
		if ((ret = encode_write(frame)) < 0) {
			fprintf(stderr ,"Error during encodeing and writing.\n");
		}

fail:
	
		av_frame_free(&frame);
		if (ret < 0) {
			return ret;
		}
	}
	return 0;
}


