#include <unistd.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>

static const char *filter_descr = "aresample=8000,aformat=sample_fmts=s16:channel_layouts=mone";
static const char *player = "ffplay -f s16le -ar 8000 -ac 1 -";

static AVFormatContext *fmt_ctx;
static AVCodecContext *dec_ctx;
AVFilterContext *buffersink_ctx;
AVFilterContext *buffersrc_ctx;
AVFilterGraph *filter_graph;
static int audio_stream_index = -1;

static int open_input_file(const char *filename) {
	int ret;
	AVCodec *dec;
	if ((ret = avformat_open_input(&fmt_ctx ,filename ,NULL ,NULL)) < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot open input file\n");
		return ret;
	}
	if ((ret = avformat_find_stream_info(fmt_ctx ,NULL)) < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot find stream information\n");
		return ret;
	}

	/** select the audio stream **/
	ret = av_find_best_stream(fmt_ctx ,AVMEDIA_TYPE_AUDIO ,-1 ,-1 ,&dec ,0);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot find an audio stream in the input file\n");
		return ret;
	}
	audio_stream_index = ret;

	/** create decoding context **/
	dec_ctx = avcodec_alloc_context3(dec);
	if (!dec_ctx) {
		return AVERROR(ENOMEM);
	}
	avcodec_parameters_to_context(dec_ctx ,fmt_ctx->streams[audio_stream_index]->codecpar);
	/** init the audio decoder **/
	if ((ret = avcodec_open2(dec_ctx ,dec ,NULL)) < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot open audio decoder\n");
		return ret;
	}

	return 0;
}

static int init_filters(const char *filters_descr) {
	char args[512];
	int ret = 0;
	const AVFilter *abuffersrc = avfilter_get_by_name("abuffer");
	const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
	AVFilterInOut *outputs = avfilter_inout_alloc();
	AVFilterInOut *inputs = avfilter_inout_alloc();
	static const enum AVSampleFormat out_sample_fmts[] = { AV_SAMPLE_FMT_S16 ,-1};
	static const int64_t out_channel_layouts[] = {8000 ,-1};
	const AVFilterLink *outlink;
	AVRational time_base = fmt_ctx->streams[audio_stream_index]->time_base;

	filter_graph = avfilter_graph_alloc();
	if (!outputs || !inputs || !filter_graph) {
		ret = AVERROR(ENOMEM);
		goto end;
	}

	/** buffer audio source : the decoded frames from the decoder will be inserted here. **/
	if (!dec_ctx->channel_layout) {
		dec_ctx->channel_layout = av_get_default_channel_layout(dec_ctx->channels);
	}
	snprintf(args ,sizeof(args) ,"time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%"PRIx64,time_base.num ,time_base.den ,dec_ctx->sample_rate ,av_get_sample_fmt_name(dec_ctx->sample_fmt) ,dec_ctx->channel_layout);
	ret = avfilter_graph_create_filter(&buffersrc_ctx ,abuffersrc ,"in" ,args ,NULL ,filter_graph);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot create audio buffer source.\n");
		goto end;
	}
	ret = av_opt_set_int_list(buffersink_ctx ,"sample_fmts" ,out_sample_fmts ,-1 ,AV_OPT_SEARCH_CHILDREN);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot set output sample format\n");
		goto end;
	}
	ret = av_opt_set_int_list(buffersink_ctx ,"sample_rates" ,out_sample_fmts ,-1 ,AV_OPT_SEARCH_CHILDREN);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot set output sample rate\n");
		goto end;
	}
	/** Set the endpoints for the filter graph. The filter_graph will be linked to the graph described by filter_descr **/
	/** The buffer source output must be connected to the input pad of the filter input label is not specified ,it is set to "in" by default **/
	outputs->name = av_strdup("in");
	outputs->filter_ctx = buffersrc_ctx;
	outputs->pad_idx = 0;
	outputs->next = NULL;

	/** The buffer sink input must be connected to the output pad of the last filter described by filters_descr; Since the last filter output label is not specified, it is set to "out" by default **/
	inputs->name = av_strdup("out");
	inputs->filter_ctx = buffersink_ctx;
	inputs->pad_idx = 0;
	inputs->next = NULL;

	if ((ret = avfilter_graph_parse_ptr(filter_graph ,filters_descr ,&inputs ,&outputs ,NULL)) < 0) {
		goto end;
	}
	if ((ret = avfilter_graph_config(filter_graph ,NULL)) < 0) {
		goto end;
	}
	/** Print summary of the sink buffer
	 *  Note : args buffer is reused to store channel layout string
	 * **/
	outlink = buffersink_ctx->inputs[0];
	av_get_channel_layout_string(args ,sizeof(args) ,-1 ,outlink->channel_layout);		av_log(NULL ,AV_LOG_INFO ,"Output : srate:%dHz fmt:%s chlayout:%s\n"				,(int)outlink->sample_rate ,(char *)av_x_if_null(av_get_sample_fmt_name(outlink->format) ,"?") ,args);
end:
	avfilter_inout_free(&inputs);
	avfilter_inout_free(&outputs);
	return ret;
}

static void print_frame(const AVFrame *frame) {
	const int n = frame->nb_samples * av_get_channel_layout_nb_channels(frame->channel_layout);
	const uint16_t *p = (uint16_t *)frame->data[0];
	const uint16_t *p_end = p + n;
	while (p < p_end) {
		fputc(*p & 0xff ,stdout);
		fputc(*p>>8 & 0xff ,stdout);
		p++;
	}
	fflush(stdout);
}

int main(int argc ,char **argv) {
	int ret;
	AVPacket packet;
	AVFrame *frame = av_frame_alloc();
	AVFrame *filter_frame = av_frame_alloc();

	if (!frame || !filter_frame) {
		perror("Could not allocate frame");
		exit(1);
	}
	if (argc != 2) {
		fprintf(stderr ,"Usage: %s file | %s \n" ,argv[0] ,player);
		exit(1);
	}
	if ((ret = open_input_file(argv[1])) < 0) {
		goto end;
	}
	if ((ret = init_filters(filter_descr)) < 0) {
		goto end;
	}
	/** read the packet **/
	while (1) {
		if (ret = av_read_frame(fmt_ctx ,&packet) < 0) {
			break;
		}
		if (packet.stream_index == audio_stream_index) {
			ret = avcodec_send_packet(dec_ctx ,&packet);
			if (ret < 0) {
				av_log(NULL ,AV_LOG_ERROR ,"Error while sending a packet to the decoder\n");
				break;
			}
			while (ret >= 0) {
				ret = avcodec_receive_frame(dec_ctx ,frame);
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				} else if (ret < 0) {
					av_log(NULL ,AV_LOG_ERROR ,"Error while receiving a frame from the decoder\n");
					goto end;
				}

				if (ret >= 0) {
					/** push the audio data from decoded frame video into the filterggraph **/
					if (av_buffersrc_add_frame_flags(buffersrc_ctx ,frame ,AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
						av_log(NULL ,AV_LOG_ERROR ,"Error while feeding the audio filtergraph\n");
						break;
					}
					/** pull filtered audio from the filtergraph **/
					while (1) {
						ret = av_buffersink_get_frame(buffersink_ctx ,filter_frame);
						if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
							break;
						}
						if (ret < 0) {
							goto end;
						}
						print_frame(filter_frame);
						av_frame_unref(filter_frame);
					}
				}
			}
		}
		av_packet_unref(&packet);
	}
end:
	avfilter_graph_free(&filter_graph);
	avcodec_free_context(&dec_ctx);
	avformat_close_input(&fmt_ctx);
	av_frame_free(&frame);
	av_frame_free(&filter_frame);
	if (ret < 0 && ret != AVERROR_EOF) {
		fprintf(stderr, "Error occurred:%s \n" ,av_err2str(ret));
		exit(1);
	}
	exit(0);
}

/**
 * 1. struct
 * 1.1 typedef struct AVFilterInOut { }AVFilterInOut;
 * 1.2 #define av_opt_set_int_list(obj ,name ,val ,term ,flags)
 *
 * 2. function
 * 2.1 (self define) static int open_input_file(const char *filename);
 * 2.2 (self define) static int init_filters(const char **filters_descr);
 * 2.3 AVFilterInOut *avfilter_inout_alloc(void);
 * 2.4 int avfilter_graph_create_filter(AVFilterContext **filt_ctx ,const AVFilter *filt ,const char *name ,const char *args ,void *opaque ,AVFilterGraph *graph_ctx);
 * 2.5 char *av_strdup(const char *s);
 * 2.6 int avfilter_graph_parse_ptr(AVfilterGraph *graph ,const char *filters ,AVFilterInOut **inputs ,AVFilterInOut **outputs ,void *log_ctx);
 * */