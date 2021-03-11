#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <stdio.h>

int interrupt_cb(void* ctx) {
	return 0;
} 

int main (int argc ,char* argvs[] ) {
	int ret;
	char* input[2];
	char* output;
	AVFormatContext* inputCtx[2];
	AVCodecContext* outputCodecCtx[2];

	if (argc != 4) {
		fprintf(stderr ,"Input Format Error.\n %s <input_video_file> <input_picture_file> <output_video_file>\n" ,argvs[0]);
		return 0;
	}
	input[0] = argvs[1];
	input[1] = argvs[2];
	output = argvs[3];
	//Open Input File
	inputCtx[0] = avformat_alloc_context();
	inputCtx[0]->interrupt_callback.callback = interrupt_cb;
	AVDictionary* format_opt = NULL;
	ret = avformat_open_input(&inputCtx[0] ,input[0] ,NULL ,&format_opt);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Input File %s Failure\n" ,input[0]);
		goto end;
	}
	ret = avformat_find_stream_info(inputCtx[0] ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Info Input File %s Failure\n" ,input[0]);
		goto end;
	}
	av_dump_format(inputCtx[0] ,0 ,input[0] ,0);
	enum AVCodecID codecId = inputCtx[0]->streams[0]->codecpar->codec_id;
	AVCodec* codec = avcodec_find_decoder(codecId);	
	if (!codec) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Video# %d failure\n" ,codecId);
		goto end;
	}
	outputCodecCtx[0] = avcodec_alloc_context3(codec);
	if (!outputCodecCtx[0]) {	
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Context Video# %d failure\n" ,codecId);
		goto end;
	}
	ret = avcodec_open2(outputCodecCtx[0] ,codec ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Output Codec For File %s Failure\n" ,input[0]);
		goto end;
	}

	inputCtx[1] = avformat_alloc_context();
	inputCtx[1]->interrupt_callback.callback = NULL;
	ret = avformat_open_input(&inputCtx[1] ,input[1] ,NULL ,&format_opt);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Input File %s Failure\n" ,input[1]);
		goto end;
	}
	ret = avformat_find_stream_info(inputCtx[1] ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Info Input File %s Failure\n" ,input[1]);
		goto end;
	}
	av_dump_format(inputCtx[1] ,0 ,input[1] ,0);
	enum AVCodecID codecIdPic = inputCtx[1]->streams[0]->codecpar->codec_id;
 	AVCodec* codecPic = avcodec_find_decoder(codecIdPic);	
	if (!codecPic) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Video# %d failure\n" ,codecIdPic);
		goto end;
	}
	outputCodecCtx[1] = avcodec_alloc_context3(codecPic);
	if (!outputCodecCtx[1]) {	
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Context Video# %d failure\n" ,codecId);
		goto end;
	}
	ret = avcodec_open2(outputCodecCtx[1] ,codecPic ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Output Codec For File %s Failure\n" ,input[1]);
		goto end;
	}

	fprintf(stderr ,"\nFRAME OVERLAY SUCCESS\n");
	return 0;
end:
	fprintf(stderr ,"\nFRAME OVERLAY FAILURE\n");
	return -1;
}
