#include "config.h"
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

	}

}
