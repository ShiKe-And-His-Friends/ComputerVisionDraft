#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>

typedef struct StreamContext {
	AVCodecContext* dec_ctx;
	AVCodecContext* enc_ctx;
}StreamContext;

int interrupt_cb(void* ctx) {
	fprintf(stderr ,"interrupt cb.\n");
	return 0;
}

int main(int argc ,char* argv[]) {
	int ret;
	char* output = NULL;
	char* input = NULL;
	AVFormatContext* context = NULL;
	AVFormatContext* output_context = NULL;
	AVDictionary* format_opts = NULL;
	AVCodec* codec = NULL;
	AVCodec* h264Codec = NULL;
	AVCodecContext* decode_ctx = NULL;
	AVCodecContext* encode_ctx = NULL;
	AVStream* in_stream = NULL;
	AVStream* out_stream = NULL;
	StreamContext* stream_ctx = NULL;

	if (argc != 3) {
		fprintf(stderr ,"input format error.\n %s <input_file> <output_file> \n" ,argv[0]);
		return -1;
	}

	input = argv[1];
	output = argv[2];

	
	context = avformat_alloc_context();
	context->interrupt_callback.callback = interrupt_cb;
	ret = avformat_open_input(&context ,input ,NULL ,&format_opts);
	if (!ret) {
		av_dump_format(context ,0 ,input ,0);
	} else {
		return 0;
	}
	stream_ctx = av_mallocz_array(context->nb_streams ,sizeof(*stream_ctx));
	ret = avformat_find_stream_info(context ,NULL);
	if (ret) {
		fprintf(stderr ,"Find stream best info failure.\n");
		return 0;
	}
	
	codec = avcodec_find_decoder(context->streams[0]->codecpar->codec_id);	
	decode_ctx = avcodec_alloc_context3(codec);
	if (!decode_ctx) {
		fprintf(stderr ,"Find stream codec failure.\n");
		return 0;
	}
	ret = avcodec_parameters_to_context(decode_ctx ,context->streams[0]->codecpar);
	if (ret < 0) {
		fprintf(stderr ,"File copy parameters failure.\n");
		return 0;
	}
	if (codec->capabilities & AV_CODEC_CAP_TRUNCATED) {
		decode_ctx->flags |= AV_CODEC_CAP_TRUNCATED;
	}
	ret = avcodec_open2(decode_ctx ,codec ,NULL);
	if (ret < 0) {
		fprintf(stderr ,"File Codec Open Failure.\n");
		return 0;
	}

	// open output file
	avformat_alloc_output_context2(&output_context ,NULL ,NULL ,output);
	if (!output_context) {
		av_log(NULL ,AV_LOG_ERROR ,"Could not open output context.\n");
		return AVERROR_UNKNOWN;
	}
	out_stream = avformat_new_stream(output_context ,NULL);
	if (!out_stream) {
		av_log(NULL ,AV_LOG_ERROR ,"Failed allocting output stream.\n");
		return AVERROR_UNKNOWN;
	}
	in_stream = context->streams[0];
	h264Codec = avcodec_find_encoder(decode_ctx->codec_id);
	if (!h264Codec) {
		av_log(NULL ,AV_LOG_ERROR ,"Necessary encoder not found.\n");
		return AVERROR_INVALIDDATA;
	}
	encode_ctx = avcodec_alloc_context3(h264Codec);
	if (!encode_ctx) {
		av_log(NULL ,AV_LOG_ERROR ,"Necessary encoder context not found.\n");
		return AVERROR_INVALIDDATA;
	}
	encode_ctx->height = decode_ctx->height;
	encode_ctx->width = decode_ctx->width;
	encode_ctx->sample_aspect_ratio = decode_ctx->sample_aspect_ratio;
	if (h264Codec->pix_fmts) {
		encode_ctx->pix_fmt = h264Codec->pix_fmts[0];
	} else {
		encode_ctx->pix_fmt = decode_ctx->pix_fmt;
	}
	encode_ctx->time_base = av_inv_q(decode_ctx->framerate);
	AVDictionary* options = NULL;
	encode_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
	ret = avcodec_open2(encode_ctx ,h264Codec ,&options);
	if (ret < 0) {
		fprintf(stderr ,"Open Out Codec Failure.\n");
		return -1;
	}

	fprintf(stderr ,"\nFINISH\n");
	return 0;
}
