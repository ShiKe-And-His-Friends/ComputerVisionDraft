#define _XOPEN_SOURCE 600 /** for thread usleep **/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <linavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <liavutil/opt.h>

const char *filter_descr = "scale=78:24,transpose=cclock"
/** other way:
 *  scale=78:24[scl]:[scl] transpose=cclock 
 *  //assumes "[in]" and "[out]" to be input output pads respectively
 * **/
static AVFormatContext *fmt_ctx;
static AVCodecContext *dec_ctx;
AVFilterContext *buffersink_ctx;
AVFilterContext *buffersrc_ctx;
AVFilterGraph *filter_graph;
static int video_stream_index = -1;
static int64_t last_pts = AV_NOPTS_VALUE;

static int open_input_file(const char *filename) {
	int ret;
	AVCodec *dec;
	if ((ret = avformat_input(&fmt_ctx ,filename ,NULL ,NULL)) < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot open input file\n");
		return ret;
	}
	if ((ret = avformat_find_stream_info(fmt_ctx ,NULL)) < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Can not find stream info \n");
		return ret;
	}

	/** select the video stream **/
	ret = av_find_best_stream(fmt_ctx ,AVMEDIA_TYPE_VIDEO ,-1 ,-1 ,&dec ,0);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot find video stream int the input file\n");
		return ret;
	}
	video_stream_index = ret;

	/** create decoding context **/
	dec_ctx = avcodec_alloc_context3(dec);
	if (!dec_ctx) {
		return AVERROR(ENOMEM);
	}
	avcodec_parameters_to_context(dec_ctx ,fmt_ctx->streams[video_stream_index]->codecpar);

	/** init the video decoder **/
	if ((ret = avcodec_open2(dec_ctx ,dec ,NULL)) < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot open video decoder\n");
		return ret;
	}
	return 0;
}

static int init_filters(const char *filters_descr) {
	char args[512];
	int ret = 0;
	const AVFilter *buffersrc = avfilter_get_by_name("buffer");
	const AVFilter *buffersink = avfilter_get_by_name("buffersink");
	AVFilterInOut *outputs = avfilter_inout_alloc();
	AVFilterInOut *inputs = avfilter_inout_alloc();
	AvRational time_base = fmt_ctx->streams[video_stream_index]->time_base;
	enum AVpixelFormat pix_fmts[] = {AV_PIX_FMT_GRAY8 ,AV_PIX_FMT_NONE};
	filter_graph = avfilter_graph_alloc();
	if (!output || !input || !filter_graph) {
		ret = AVERROR(ENOMEM);
		goto end;
	}

	/** buffer video source : the decoded frames from the decoders will be inserted here **/
	snprintf(args ,sizeof(args) ,"video_size=%dx%s:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d" ,dec_ctx->width ,dec_ctx->height ,dex_ctx->pix_fmt ,time_base.num ,time_base.den ,dec_ctx->sample_aspect_ratio.num ,dec_ctx->sample_aspect_ratio.den);


	ret = avfilter_graph_filter(&buffersrc_ctx ,buffersrc ,"in"
			,args ,NULL ,filter_graph);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot create buffer source\n");
		goto end;
	}
	/** buffer video sink: to terminate the filter chain. **/
	ret = avfilter_graph_create_filter(&buffersink_ctx ,buffersink ,"out"
			,NULL ,NULL ,filter_graph);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot create buffer sink\n");
		goto end;
	}

	ret = av_opt_set_int_list(buffersink_ctx ,"pix_fmts" ,pix_fmts
			,AV_PIX_FMT_NONE ,AV_OPT_SEARCH_CHILDREN);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Cannot set output pixel format\n");
		goto end;
	}
	/** Set the endpoints for the filter graph.The filter_graph will be linked to the graph described by filters_descr **/
	/** The buffer source output must be connected to the input pad of the first filter described by filters_descr;
	 *  Since the first filter input label is not specified ,it is set to "in" by default **/
	output->name = av_strdup("in");
	output->filter_ctx = buffersink_ctx;
	output->pad_idx = 0;
	output->next = NULL;

	/** The buffer sink input must be connected to the output pad of the last fiter described by fiters_descr;
	 *  Since the last filter output label is not specified ,it is set to "out" by default **/
	input->name = av_strdup("out");
	input->filter_ctx = buffersink_ctx;
	input->pad_idx = 0;
	input->next = NULL;
	if ((ret = avfilter_graph_parse_ptr(filter_graph ,filters_descr ,&inputs ,&outputs ,NULL)) < 0) {
		goto end;
	}
	if ((ret = avfilter_graph_config(filter_graph ,NULL)) < 0) {
		goto end;
	}
end:
	avfilter_inout_free(&inputs);
	avfilter_inout_free(&outputs);

	return ret;
}


