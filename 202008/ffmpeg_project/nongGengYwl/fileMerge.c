#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <stdio.h>

int main (int argc ,char* argvs[]) {
	uint8_t index = 0;
	int ret;
	int size;
	int lastPacketPts = 0;
	int lastPts = 0;
	char* input[2];
	char* output;
	AVFormatContext* inputCtx[2];
	AVFormatContext* outputCtx;
	AVPacket packet;
	if (argc != 4) {
		fprintf(stderr ,"Format Error.\n %s <input_file1> <input_file2> <output_file>\n" ,argvs[0]);
		return -1;
	}
	input[0] = argvs[1];
	input[1] = argvs[2];
	output = argvs[3]; 
	size = (sizeof(inputCtx) / (sizeof(AVFormatContext *)));

	//Open input files
	inputCtx[0] = avformat_alloc_context();
	inputCtx[1] = avformat_alloc_context();
	inputCtx[0]->interrupt_callback.callback = NULL;
	AVDictionary* options = NULL;
	ret = avformat_open_input(&inputCtx[0] ,input[0] ,NULL,&options);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Input %s Context Failure\n" ,input[0]);
		goto end;
	}
	ret = avformat_find_stream_info(inputCtx[0] ,NULL);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Input %s Info Failure\n" ,input[0]);
		goto end;
	} else {
		av_log(NULL ,AV_LOG_INFO ,"Open Success 1.\n");
	}
	
	inputCtx[1]->interrupt_callback.callback = NULL;
	ret = avformat_open_input(&inputCtx[1] ,input[1] ,NULL,&options);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Input %s Context Failure\n" ,input[1]);
		goto end;
	}
	ret = avformat_find_stream_info(inputCtx[1] ,NULL);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Input %s Info Failure\n" ,input[1]);
		goto end;
	} else {
		av_log(NULL ,AV_LOG_INFO ,"Open Success 2.\n");
	}
	
	//Open Output File
	ret = avformat_alloc_output_context2(&outputCtx ,NULL ,"mp4" ,output);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Output %s Context Failure\n" ,output);
		goto end;
	}
	ret = avio_open2(&outputCtx->pb ,output ,AVIO_FLAG_WRITE ,NULL ,NULL);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Output Stream %s Failure\n" ,output);
		goto end;
	}
	for (int i = 0 ; i < inputCtx[0]->nb_streams ;i++) {
		AVStream* stream = avformat_new_stream(outputCtx ,NULL);
		avcodec_parameters_copy(stream->codecpar ,inputCtx[i]->streams[i]->codecpar);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Open Output Stream#%u Failure\n" ,i);
			goto end;
		}
	}
	ret = avformat_write_header(outputCtx ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Write Output File Failure\n");
		goto end;
	}
	av_log(NULL ,AV_LOG_INFO ,"Open Output File Success\n");
	
	// Process Data
	while(1) {
		packet.size = 0;
		packet.data = NULL;
		av_init_packet(&packet);
		ret = av_read_frame(inputCtx[index] ,&packet);
		if (ret >= 0) {
			int64_t diff = packet.pts - lastPacketPts;
//			av_log(NULL ,AV_LOG_INFO ,"index = %d diff = %d packet.pts = %ls \nlastPacketPts = %ld\n\n\n" ,index ,diff ,packet.pts ,lastPacketPts);
			if (diff > 0) {
				lastPts += diff;
			} else {
				lastPts += 1;
			}
			lastPacketPts = packet.pts;
			packet.pts = packet.dts = lastPts;
			av_log(NULL ,AV_LOG_INFO ,"pts dts = %d  lastPackFlag = %d\n" ,lastPts ,lastPacketPts);
			AVStream* inputStream = inputCtx[index]->streams[packet.stream_index];
			AVStream* outputStream = outputCtx->streams[packet.stream_index];
			av_packet_rescale_ts(&packet ,inputStream->time_base ,outputStream->time_base);
			ret = av_interleaved_write_frame(outputCtx ,&packet);
			av_packet_unref(&packet);
			if (ret < 0) {
				av_log(NULL ,AV_LOG_ERROR ,"Packet Write Failure.\n");
				//check line
				goto end;
			}
		} else {
			av_packet_unref(&packet);
			index ++;
			if (index == size) {
				av_log(NULL ,AV_LOG_INFO ,"\nFile End...\n");
				break;
			}
		}
	}

	av_write_trailer(outputCtx);
	avformat_free_context(outputCtx);
	avformat_free_context(inputCtx[0]);
	avformat_free_context(inputCtx[1]);
	fprintf(stderr ,"\n\nMERGE SUCCESS\n");
	return 0;

end:
	avformat_close_input(&outputCtx);
	fprintf(stderr ,"MERGE FAILURE\n");
	return -1;
}
