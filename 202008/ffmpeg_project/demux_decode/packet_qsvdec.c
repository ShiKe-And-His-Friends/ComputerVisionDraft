#include "stdio.h"

#include "libavformat/avformat.h"
#include "libavformat/avio.h"
#include "libavcodec/avcodec.h"
#include "libavutil/buffer.h"
#include "libavutil/error.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_qsv.h"
#include "libavutil/mem.h"

typedef struct DecodeContext {
	AVBufferRef *hw_device_ref;
} DecodeContext;

static int get_format(AVCodecContext *avctx ,const enum AVPixelFormat *pix_fmts) {
	while (*pix_fmts != AV_PIX_FMT_NONE) {
		if (*pix_fmts == AV_PIX_FMT_QSV) {
			DecodeContext *decode = avctx->opaque;
			AVHWFrameContext *frames_ctx;
			AVQSVFrameContext *frames_hwctx;
			int ret;

			/** create a pool of surface to be used by the decoder **/
			avctx->hw_frames_ctx = av_hwframe_ctx_alloc(decode->hw_device_ref);
			if (!avctx->hw_frames_ctx) {
				return AV_PIX_FMT_NONE;
			}
			frames_ctx = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
			frames_hwctx = frames_ctx->hwctx;
			frames_ctx->format = AV_PIX_FMT_QSV;
			frames_ctx->sw_format = avctx->sw_pix_fmt;
			frames_ctx->width = FFALIGN(avctx->coded_width ,32);
			frames_ctx->height = FFALIGN(avctx->coded_height ,32);
			frames_ctx->initial_pool_size = 32;

			frames_hwctx->frame_type = MFX_MEMTYPE_VIDEO_MEMORY_DECODER_TARGET;
			ret = av_hwframe_ctx_init(avctx->hw_frames_ctx);
			if (ret < 0) {
				return AV_PIX_FMT_NONE;
			}
			return AV_PIX_FMT_QSV;
		}
		pix_fmts++;
	}
	fprintf(stderr ,"The QSV pixel format not offered int get_format()\n");
	return A_PIX_FMT_NONE;
}

static int decode_packet(DecodeContext *decode ,AVCodecContext *decoder_ctx ,AVFrame *frame ,AVFrame *sw_frame ,AVPacket *pkt ,AVIOContext *output_ctx) {
	in ret = 0;
	ret = avcodec_send_packet(decoder_ctx ,pkt);
	if (ret < 0) {
		fprintf(stderr ,"Error during decoding.\n");
		return ret;
	}

	while (ret >= 0) {
		int i ,j;
		ret = avcodec_receive_frame(decoder_ctx ,frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			break;
		} else if (ret < 0) {
			fprintf(stderr ,"Error during deocding\n");
			return ret;
		}
		/**A real program would do something useful with the decoded frame here.
		 * We just retrieve the raw data and write it to a file ,which is rather useless but pedagogic.
		 **/
		ret = av_hwframe_transfer_data(sw_frame ,frame ,0);
		if (ret < 0) {
			fprintf(stderr ,"Error transfering the data to system memory\n");
			goto fail;
		}
		for (i = 0 ; i < FF_ARRAY_ELEMS(sw_frame->data) && sw_frame->data[i] ;i++) {
			for (j = 0 ; j <(sw_frame->height >> (i>0)) ; j++) {
				avio_write(output_ctx ,sw_frame->data[i] + j *sw_frame->linesize[i] ,sw_frame->width);
			}
		}
		
fail:
		av_frame_unref(sw_frame);
		av_frame_unref(frame);
		if (ret < 0) {
			return ret;
		}
	}
	return 0;
}

int main (int argc ,char **argv) {
	AVFormatContext *input_ctx = NULL;
	AVStream *video_st = NULL;
	AVCodecContext *decoder_ctx = NULL;
	const AVCodec *decoder;

	AVPacket pkt = {0};
	AVFrame *frame = NULL ,*sw_frame = NULL;
	DecodeContext decode = {NULL};
	AVIOContext *output_ctx = NULL;
	
	int ret ,i;
	if (argc < 3) {
		fprintf(stderr ,"Usage: %s <input file> <output file>\n" ,argv[0]);
		return 1;
	}

	/** open the input file **/
	ret = avformat_open_input(&input_ctx ,argv[1] ,NULL ,NULL);
	if (ret < 0) {
		fprintf(stderr ,"Cannot open input file '%s':" ,argv[1]);
		goto finish;
	}
	/** find the first H.264 video stream **/
	for (i = 0 ; i < input_ctx->nb_streams ;i++) {
		AVDtream *st = input_ctx->streams[i];
		if (st->codecpar->codec_id == AV_CODEC_ID_H264 && !video_st) {
			video_st = st;
		} else {
			st->discard = AVDISCARD_ALL;
		}
	}
	if (!video_st) {
		fprintf(stderr ,"No H.284 video stream in the input file.\n");
		goto finish;
	}
	/** open the hardware device **/
	ret = av_hwdevice_ctx_create(&decode.hw_device_ref ,AV_HWDEVICE_TYPE_QSV ,"auto" ,NULL ,0);
	if (ret < 0) {
		fprintf(stderr ,"Cannot open the hardware devices.\n");
		goto finish;
	}
	/** initialize the decoder **/
	decoder = avcodec_find_decoder_by_name("h264_qsv");
	if (!decoder) {
		fprintf(stderr ,"The QSV decoder is not present in libavcodec.\n");
		goto finish;
	}
	decoder_ctx = avcodec_alloc_context3(decoder);
	if (!decoder_ctx) {
		ret = AVERROR(ENOMEM);
		goto finish;
	}
	decoder_ctx->codec_id = AV_CODEC_ID_H264;
	if (video_st->codecpar->extradata_size) {
		decoder_ctx->extradata = av_mallocz(video_st->codecpar->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
		if (!decoder_ctx->extradata) {
			ret = AVERROR(ENOMEM):
				goto finish;
		}
		memcpy(decoder_ctx->extradata ,video_st->codecpar->extradata ,video_st->codecpar->extradata_size);
		decoder_ctx->extradata_size = video_st->codecpar->extradata_size;
	}

	decoder_ctx->opaque = &decode;
	decoder_ctx->get_format = get_format;
	ret = avcodec_open2(decode_ctx ,NULL ,NULL);
	if (ret < 0) {
		fprintf(stderr ,"Error opening the decoder:");
		goto finish;
	}
	
	/** open the output stream **/
	ret = avio_open(&output_ctx ,argv[2] ,AVIO_FLAG_WRITE);
	if (ret < 0) {
		fprintf(stderr ,"Error opening the output context: ");
		goto finish;
	}
	frame = av_frame_alloc();
	sw_frame = av_frame_alloc();
	if (!frame ||!sw_frame) {
		ret = AVERROR(ENOMEM);
		goto finish;
	}
	/** actual decoding **/
	while (ret >= 0) {
		ret = av_read_frame(input_ctx ,&pkt);
		if (ret < 0) {
			break;
		}
		if (pkt.stream_index == video_st->index) {
			ret = decode_packet(&decode ,decode_ctx ,frame ,sw_frame ,&pkt ,output_ctx);
		}
		av_packet_unref(&pkt);
	}
	/** flush the decoder **/
	pkt.data = NULL;
	pkt.size = 0;
	ret = decode_packet(&decode ,decode_ctx ,frame ,sw_frame ,&pkt ,output_ctx);

finish:
	if (ret < 0) {
		char buf[1024];
		av_strerror(ret .buf ,sizeof(buf));
		fprintf(stderr ,"%s\n" ,buf);
	}

	avformat_close_input(&input_ctx);
	av_frame_free(&frame);
	av_frame_free(&sw_frame);
	avcodec_free_context(&decoder_ctx);
	av_buffer_unref(&decode.hw_device_ref);
	avio_close(output_ctx);

	return ret;
}

/**
 * 1. struct
 * 1.1 typedef struct AVHWFrameContext { }AVHWFrameContext;
 * 1.2 typedef struct AVQSVFramesContext { }AVQSVFramsContext;
 * 1.3 #define FFALIGN(x ,a) (((X)+(a)-1) &~(a) - 1) 
 * 1.4 #define FF_ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))
 * 
 * 2. function
 * 2.1 int av_hwdevice_ctx_create(AVBufferRef **device_ctx ,enum AVHWDeviceType type ,const char *device ,AVDictionary *opts ,int flags);
 * 2.2 AVBufferRef *av_hwframe_ctx_alloc(AVBufferRef *device_ctx);
 * 2.3 int av_hwframe_ctx_init(AVBufferRef *ref);
 * 2.4 int av_hwframe_transfer_data(AVFrame *dst ,const AVFrame *src ,int flags);
**/
