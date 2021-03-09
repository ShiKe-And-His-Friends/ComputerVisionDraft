#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/time.h>

int64_t lastTime = -1;
static int interrupt_cb_cut(void *ctx){
	// Internet stream no response
	if (av_gettime() - lastTime > 10 * 1000 * 1000) {
		av_log(NULL ,AV_LOG_INFO ,"Interrupt\n");
		return -1;		
	} else {
		return 0;
	}
	return 0;
}

int main(int argc ,char* argv[]) {
	int ret;
	char* input;
	char* output;
	uint64_t firstPts = -1;
	AVFormatContext* inputCtx;
	AVFormatContext* outputCtx;
	AVPacket packet;

	if (argc < 3) {
		fprintf(stderr ,"format error.\n %s <input_file> <output_file> \n" ,argv[0]);
		return -1;
	}
	input  = argv[1];
	output = argv[2];
	inputCtx = avformat_alloc_context();
	inputCtx->interrupt_callback.callback = interrupt_cb_cut;
	AVDictionary* options = NULL;
	lastTime = av_gettime();
	ret = avformat_open_input(&inputCtx ,input ,NULL ,&options);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"open input context failure.\n");
		goto end;
	}
	ret = avformat_find_stream_info(inputCtx ,NULL);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"find stream info failure.\n");
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
	for (int i = 0 ; i < inputCtx->nb_streams ; i++) {
		AVStream* stream = avformat_new_stream(outputCtx ,NULL);
		avcodec_parameters_copy(stream->codecpar ,inputCtx->streams[i]->codecpar);

		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"open stream#%u failure.\n", i);
			goto end;
		}
	}
	ret = avformat_write_header(outputCtx ,NULL);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"write header failure.\n");
		goto end;
	}
	while(1) {
		lastTime = av_gettime();
		packet.data = NULL;
		packet.size = 0;
		av_init_packet(&packet);
		ret = av_read_frame(inputCtx ,&packet);

		if (ret >= 0) {
			if(firstPts < 0) {
				firstPts = packet.pts;
			}
			if (firstPts >= 0) {
				if (packet.pts - firstPts >= 3600 * 25 * 10) {
					fprintf(stderr ,"\nTIME OUT\n");
					break;
				}
			}
			AVStream* inputStream = inputCtx->streams[packet.stream_index];
			AVStream* outputStream = outputCtx->streams[packet.stream_index];
			av_packet_rescale_ts(&packet ,inputStream->time_base ,outputStream->time_base);
			ret = av_interleaved_write_frame(outputCtx ,&packet);
			av_packet_unref(&packet);	
			if (ret < 0) {
				av_log(NULL ,AV_LOG_ERROR ,"open stream#%u  failure.\n", packet.stream_index);
				goto end;
			} else {
				av_log(NULL ,AV_LOG_DEBUG ,"open stream#%u success\n" ,packet.stream_index);
			}
		} else {
			av_packet_unref(&packet);
			break;
		}
	}

	av_write_trailer(outputCtx);
	avformat_close_input(&outputCtx);
	avformat_free_context(outputCtx);
	fprintf(stderr ,"SUCCESS FILE CUT\n");
	return 0;

end:
	fprintf(stderr ,"ERROR FILE CUT\n");
	return -1;
}

