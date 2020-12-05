#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>

int main (int argc ,char **argv) {
	int ret;
	int stream_index = 0;
	int stream_mapping_size = 0;
	int *stream_mapping = NULL;
	const char *input_file_name ,*output_file_name;
	AVOutputFormat *ofmt = NULL;
	AVFormatContext *ifmt_ctx = NULL ,*ofmt_ctx = NULL;
	if (argc < 3) {
		printf("Usage: %s intput output\n" ,argv[0]);
		return 1;
	}
	input_file_name = argv[1];
	output_file_name = argv[2];
	if ((ret = avformat_open_input(&ifmt_ctx ,input_file_name ,0 ,0)) < 0) {
		fprintf(stderr ,"Could not open input file %s" ,input_file_name);
		goto end;
	}
	if ((ret = avformat_find_stream_info(ifmt_ctx ,0)) < 0) {
		fprintf(stderr ,"Failed to retrieve input stream information");
		goto end;
	}
	av_dump_format(ifmt_ctx ,0 ,input_file_name ,0);
	avformat_alloc_output_context2(&ofmt_ctx ,NULL ,"flv" ,output_file_name);
	if (!ofmt_ctx) {
		fprintf(stderr ,"Could not create output context.\n");
		ret = AVERROR_UNKNOWN;
		goto end;
	}
	stream_mapping_size = ifmt_ctx->nb_streams;
	stream_mapping = av_mallocz_array(stream_mapping_size ,sizeof(stream_mapping));
	if(!stream_mapping) {
		ret = AVERROR(ENOMEM);
		goto end;
	}
	ofmt = ofmt_ctx->oformat;
	printf("shikeDebug... 1=%d 2=%d 3=%d \n" ,(!0) ,(!-1) ,(!1));
	for (int i=0 ; i < ifmt_ctx->nb_stream ; i++ ) {
		AVStream *output_stream = NULL;
		AVStream *input_stream = ifmt_ctx->stream[i];
		AVCodecParameters *input_codecpar = ifmt_ctx->codecpar;
		if (input_codecpar->codec_type != AVMEDIA_TYPE_AUDIO && input_codecpar->codec_type != AVMDEIA_TYPE_VIDEO && input_codecpar->codec_type != AVMDEIA_TYPE_SUBTITLE) {
			stream_mapping[i] = -1;
			continue;
		}
		stream_mapping[i] = stream_index++;
		output_stream = avformat_new_stream(ofmt_ctx ,NULL);
		if (!output_stream) {
			fprintf(stderr ,"Failed allocating outputs stream.\n");
			ret = AVERROR_UNKNOWN;
			goto end;
		}
		ret = avcodec_paramters_copy(output_stream->codecpar ,input_codecpar);
		output_stream->codecpar->codec_tag = 0;
	}
	av_dump_format(ofmt_ctx ,0 ,output_file_name ,1);
	printf("shikeDebug... ret=%d \n" ,ret);
	
end:
	avformat_close_input(&ifmt_ctx);
	
}