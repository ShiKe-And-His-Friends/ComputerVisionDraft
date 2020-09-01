#include <libavutil/imgutils.h>
#include <libavutil/smaplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>

static AVFormatContext *fmt_ctx = NULL;

static int open_codec_context(int *stream_idx ,AVCodecContext **dec_ctx 
		,AVFormatContext *fmt_ctx ,enum AVMediaType type) {
	int ret ,stream_index;
	AVStream *st;
	AVDictionary *opts = NULL;

	ret = av_find_best_stream(fmt_ctx ,type ,-1 ,-1 ,NULL ,0);
	if (ret < 0) {
		fprintf(stderr ,"Could not find %d stream in input file %s",
				av_get_media_type_string(type) ,src_filename);
		return ret;
	} else {
		stream_index = ret;
		st = fmt_ctx->streams[stream_index];
		/** find decoder for the stream **/
		dec = avcodec_find_decoder(st->codecpar->codec_id);
		if (!dec) {
			fprintf(stderr ,"Failed to find %s codec\n"
					,av_get_media_type_string(type));
			return AVERROR(EINVAL);
		}

		/** Allocate a codec context for the decoder **/
		*dec_ctx = avcodec_alloc_context3(dec);
		if (!*dec_ctx) {
			fprintf(stderr ,"Failed to allocate the %s codec context\n"
					,av_get_media_type_string(type));
			return AVERROR(ENOMEN);
		}

		/** Copy codec parameters from input stream to output codec context **/
		if ((ret = avcodec_parameters_to_context(*dec_ctx ,st->codecpar)) < 0) {
			fprintf(stderr ,"Failed to copy %s codec parameters to decoder context\n"
					,av_get_media_type_string(type));
			return ret;
		}

		/** Init the decoders ,with or without reference counting **/
		av_dict_set(&opts ,"refcounted_frames" ,refcount ? "1" :"0" ,0);
		if ((ret = avcodec_open2(*dec_ctx ,dec ,&opts)) < 0) {
			fprintf(stderr ,"Failed to open %s codec\n"
					,av_get_media_type_string(type));
			return ret;
		}
		*stream_idx = stream_index;
	}
	return 0;
}
