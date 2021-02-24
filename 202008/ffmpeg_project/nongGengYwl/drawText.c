#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>

int interrupt_cb(void* ctx) {
	fprintf(stderr ,"interrupt cb.\n");
	return 0;
}

int main(int argc ,char* argv[]) {
	int ret;
	char* output = NULL;
	char* input = NULL;
	AVFormatContext* context = NULL;
	AVDictionary* format_opts = NULL;
	AVCodec* codec = NULL;
	AVCodec* h264Codec = NULL;
	AVCodecContext* decode_ctx = NULL;
	AVCodecContext* encode_ctx = NULL;

	if (argc != 3) {
		fprintf(stderr ,"input format error.\n %s <input_file> <output_file> \n" ,argv[0]);
		return -1;
	}

	input = argv[1];
	output = argv[2];

	// fprintf(stderr ,"%s %s \n" ,input ,output);
	context = avformat_alloc_context();
	context->interrupt_callback.callback = interrupt_cb;
	ret = avformat_open_input(&context ,input ,NULL ,&format_opts);
	if (!ret) {
		av_dump_format(context ,0 ,input ,0);
	} else {
		return 0;
	}

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

	if (codec->capabilities & AV_CODEC_CAP_TRUNCATED) {
		decode_ctx->flags |= AV_CODEC_CAP_TRUNCATED;
	}
	ret = avcodec_open2(decode_ctx ,codec ,NULL);
	if (ret < 0) {
		fprintf(stderr ,"File Codec Open Failure.\n");
		return 0;
	}
	h264Codec = avcodec_find_encoder(AV_CODEC_ID_H264);
	if (!h264Codec) {
		fprintf(stderr ,"Encoder Open Failure.\n");
		return 0;
	}
	encode_ctx = avcodec_alloc_context3(h264Codec);
	encode_ctx->gop_size = 30;
	encode_ctx->has_b_frames = 0;
	encode_ctx->max_b_frames = 0;
	encode_ctx->codec_id = h264Codec->id;
	encode_ctx->time_base.num = decode_ctx->time_base.num;
	encode_ctx->time_base.den = decode_ctx->time_base.den;
	encode_ctx->pix_fmt = *h264Codec->pix_fmts;
	encode_ctx->width = decode_ctx->width;
	encode_ctx->height = decode_ctx->height;
	encode_ctx->me_subpel_quality = 0;
	encode_ctx->refs = 1;
	encode_ctx->scenechange_threshold = 0;
	encode_ctx->trellis = 0;
	AVDictionary* options = NULL;	

	fprintf(stderr ,"\nFINISH\n");
	return 0;
}
