#include <libavutil/timestamp.h>
#include <libavutil/time.h>
#include <libavformat/avformat.h>

static void log_packet(const AVFormatContext *fmt_ctx ,const AVPacket *pkt ,const char *tag) {
	AVRational *time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;
	printf("%s: pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d \n" 
		,tag ,av_ts2str(pkt->pts) ,av_ts2timestr(pkt->pts ,time_base) ,av_ts2str(pkt->dts) ,av_ts2timestr(pkt->dts ,time_base)
		,av_ts2str(pkt->duration) ,av_ts2timestr(pkt->duration ,time_base) ,pkt->stream_index);
}

int main (int argc ,char **argv) {
	int ret;
	int stream_index = 0;
	int stream_mapping_size = 0;
	int64_t start_time = 0;
	int *stream_mapping = NULL;
	const char *input_file_name ,*output_file_name;
	AVPacket pkt;
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
	for (int i=0 ; i < ifmt_ctx->nb_streams ; i++ ) {
		AVStream *output_stream = NULL;
		AVStream *input_stream = ifmt_ctx->streams[i];
		AVCodecParameters *input_codecpar = input_stream->codecpar;
		if (input_codecpar->codec_type != AVMEDIA_TYPE_AUDIO && input_codecpar->codec_type != AVMEDIA_TYPE_VIDEO && input_codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE) {
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
		ret = avcodec_parameters_copy(output_stream->codecpar ,input_codecpar);
		output_stream->codecpar->codec_tag = 0;
		out_stream->codec->codec_tag = 0;
		if (octx->oformat->flags & AVFMT_GLOBALHEADER) {
            out_stream->codec->flags = out_stream->codec->flags | CODEC_FLAG_GLOBAL_HEADER;
        }
	}
	av_dump_format(ofmt_ctx ,0 ,output_file_name ,1);
	if (!(ofmt->flags & AVFMT_NOFILE)) {
		ret = avio_open(&ofmt_ctx->pb ,output_file_name ,AVIO_FLAG_WRITE);
		if (ret < 0) {
			fprintf(stderr ,"Could not open output file stream '%s' " ,output_file_name);
			goto end;
		}
	} else {
		fprintf(stderr ,"Failed find output file.\n");
		goto end;
	}
	
	/**
		ADD SOME DEBUG CODE
		Failed to update header with correct duration.
		AVDictionary * opts = nullptr;
	    	av_dict_set(&opts, "flvflags", "no_duration_filesize", 0);

		ret = avformat_write_header(ofmt_ctx ,pts ? &opts : NULL);
		AVDictionary * opts = NULL;
		av_dict_set(&opts, "flvflags", "no_duration_filesize", 0);
		ret = avformat_write_header(ofmt_ctx ,&opts);
	**/

	ret = avformat_write_header(ofmt_ctx ,NULL);
	if (ret < 0) {
		fprintf(stderr ,"Error occured when opeing output file.\n");
		goto end;
	}
	
	start_time = av_gettime();
	while(1) {
		AVStream *in_stream ,*out_stream;
		ret = av_read_frame(ifmt_ctx ,&pkt);
		if (ret < 0) {
			printf("\n stream finished... \n");
			break;
		}
		in_stream = ifmt_ctx->streams[pkt.stream_index];
		if (pkt.stream_index >= stream_mapping_size || stream_mapping[pkt.stream_index] < 0) {
			av_packet_unref(&pkt);
			continue;
		}
		pkt.stream_index = stream_mapping[pkt.stream_index];
		out_stream = ofmt_ctx->streams[pkt.stream_index];
		log_packet(ifmt_ctx ,&pkt ,"in");
		
		 //延时
        if (pkt.stream_index == AVMEDIA_TYPE_VIDEO) {
            AVRational time_base = ictx->streams[videoindex]->time_base;
            AVRational time_base_q = { 1,AV_TIME_BASE };
            //计算视频播放时间
            int64_t pts_time = av_rescale_q(pkt.dts, time_base, time_base_q);
            //计算实际视频的播放时间
            int64_t now_time = av_gettime() - start_time;

            AVRational avr = ictx->streams[videoindex]->time_base;
            cout << avr.num << " " << avr.den << "  "<<pkt.dts <<"  "<<pkt.pts<<"   "<< pts_time <<endl;
            if (pts_time > now_time) {
                //睡眠一段时间（目的是让当前视频记录的播放时间与实际时间同步）
                av_usleep((unsigned int)(pts_time - now_time));
            }
        }
		
		/** copy packet **/
		pkt.pts = av_rescale_q_rnd(pkt.pts ,in_stream->time_base ,out_stream->time_base ,AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
		pkt.dts = av_rescale_q_rnd(pkt.dts ,in_stream->time_base ,out_stream->time_base ,AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);
		pkt.duration = av_rescale_q(pkt.duration ,in_stream->time_base ,out_stream->time_base);
		pkt.pos = -1;
		log_packet(ofmt_ctx ,&pkt ,"out");

		ret = av_interleaved_write_frame(ofmt_ctx ,&pkt);
		if (ret < 0) {
			fprintf(stderr ,"Error muxing packet.\n");
			break;
		}
		av_packet_unref(&pkt);
	}
	av_write_trailer(ofmt_ctx);
	
	printf("\n\nshikeDebug... ret=%d \n\n" ,ret);
	
end:
	avformat_close_input(&ifmt_ctx);
	if (ofmt_ctx && !(ofmt->flags & AVFMT_NOFILE)) {
		avio_closep(&ofmt_ctx->pb);
	}
	avformat_free_context(ofmt_ctx);
	av_freep(&stream_mapping);
	if (ret < 0 && ret != AVERROR_EOF) {
		fprintf(stderr ,"Error occurred: %s \n" ,av_err2str(ret));
		return 1;
	}
	return 0;
}
