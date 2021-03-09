#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/time.h>

int64_t lastTime;
static int interrupt_cb_cut(void *ctx){
	if (av_gettime() - lastTime > 10 * 1000 * 1000) {
		return -1;		
	} else {
		return 0;
	}
}

int main(int argc ,char* argv[]) {
	int ret;
	char* input;
	char* output;
	AVFormatContext* inputCtx;
	AVFormatContext* outputCtx;

	if (argc < 3) {
		fprintf(stderr ,"format error.\n %s <input_file> <output_file> \n" ,argv[0]);
		return -1;
	}
	input  = argv[1];
	output = argv[2];
	inputCtx = avformat_alloc_context();
	lastTime = av_gettime();
	inputCtx->interrupt_callback.callback = interrupt_cb_cut;
	AVDictionary* options = NULL;
	ret = avformat_open_input(&inputCtx ,input ,NULL ,&options);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"open input context failure.\n");
		goto end;
	}
	avformat_alloc_output_context2(&outputCtx ,NULL ,"mp4" ,output);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"open output  context failure.\n");
		goto end;
	}
	ret = avio_open2(&outputCtx->pb ,output ,AVIO_FLAG_WRITE ,NULL ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"open output file failure.\n");
		goto end;
	}
	

	fprintf(stderr ,"SUCCESS FILE CUT\n");
	return 0;

end:
	fprintf(stderr ,"ERROR FILE CUT\n");
	return -1;
}

