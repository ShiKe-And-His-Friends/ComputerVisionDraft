#include <inttypes.h>
#include <math.h>
#include <stdin.h>
#include <stdlib.h>

#include "libavutil/channel_layout.h"
#include "libavutil/md5.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/samplefmt.h"

#include "libavfilter/avfilter.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"

#define INPUT_SAMPLERATE 48000
#define INPUT_FORMAT AV_SAMPLE_FMT_FLTP
#define INPUT_CHANNEL_LAYOUT AV_CH_LAYOUT_SPOINT0

#define VOLUME_VAL 0.90

static int init_filter_graph(AVFilterGraph **graph ,AVFilterContext **src ,AVFIlterContext **sink) {
	AVFilterGraph *filter_graph;
	AVFilterContext *abuffer_ctx;
	const AVFilter *abuffer;
	AVFilterContext *volume_ctx;
	const AVFilter *volume;
	AVFilterContext *aformat_ctx;
	const AVFilter *aformat;
	AVFilterContext *abuffersink_ctx;
	const AVFilter *abuffersink;

	AVDictionary *options_dist = NULL;
	uint8_t options_str[1024];
	uint8_t ch_layout[64];

	int err;

	/** Create a new filtergraph ,which will contain all the filters. **/
	filter_graph = avfilter_graph_alloc();
	if (!filter_graph) {
		fprintf(stderr, "Unable to create filter graph.\n");
		return AVERROR(ENOMEM);
	}

	/** Create the abuffer filter. It will be used for feeding the data into the graph. **/
	abuffer = avfilter_get_by_name("abuffer");
	if (!abuffer) {
		fprintf(stderr ,"Could not find the abuffer filter.\n");
		return AVERROR_FILTER_NOT_FOUND;
	}
	abuffer_ctx = avfilter_graph_alloc_filter(filter_graph ,abuffer ,"src");
	if (!abuffer_ctx) {
		fprintf(stderr ,"Could not allocate the abuffer instance.\n");
		return AVERROR(ENOMEM);
	}
	/** Set the filter options through the AVOptions API. **/
	av_get_channel_layout_string(ch_layout ,sizeof(ch_layout) ,0 ,INPUT_CHANNEL_LAYOUT);
	av_opt_set(abuffer_ctx ,"channel_layout" ,ch_layout ,AV_OPT_SEARCH_CHILDREN);
	av_opt_set(abuffer_ctx ,"sample_fmt" ,av_get_sample_fmt_name(INPUT_FORMAT) ,AV_OPT_SEARCH_CHILDREN);
	av_opt_set_q(abuffer_ctx ,"time_base" , (AVRational){1 ,INPUT_SAMPLERATE} ,AV_OPT_SEARCH_CHILDREN);
	av_opt_set_int(abuffer_ctx ,"sample_rate" ,INPUT_SAMPLERATE ,AV_OPT_SEARCH_CHILDREN);

	/** Now initialize the filter; we pass NULL options ,since we have already set all options above **/
	err = avfilter_init_str(abuffer_ctx ,NULL);
	if (err < 0) {
		fprintf(stderr ,"Could not initialize the abuffer filter\n");
		return err;
	}

	/** Create volume filter **/
	volume = avfilter_get_by_name("volume");
	if (!volume) {
		fprintf(stderr ,"Could not find the volume filter.\n");
		return AVERROR_FILTER_NOT_FOUND;
	}

	volume_ctx = avfilter_graph_alloc_filter(filter_graph ,volume ,"volume");
	if (!volume_ctx) {
		fprintf(stderr ,"Could not allocate the volume instance.\n");
		return AVERROR(ENOMEM);
	}

	/** A different way of passing the options is as key/value pairs in a dictionary. **/
	av_dict_set(&options_dist ,"volume" ,AV_STRINGIFY(VOLUME_VAL) ,0);
	err = avfilter_init_dict(volume_ctx ,&options_dist);
	av_dict_free(&options_dist);
	if (err < 0) {
		fprintf(stderr ,"Could not initialize the volume filter.\n");
		return err;
	}

	/** Create the aformat filter; it ensures that the output is of the format we want **/
	aformat = avfilter_get_by_name("aformat");
	if (!aformat) {
		fprintf(stderr ,"Could not find the aformat filter.\n");
		return AVERROR_FILTER_NOT_FOUND;
	}
	aformat_ctx = avfilter_graph_alloc_filter(filter_graph ,aformat ,"aformat");
	if (!aformat_ctx) {
		fprintf(stderr ,"Could not allocate the aformat instance.\n");
		return AVERROR(ENOMEM);
	}

	/** A third way of passing the options is in a string of the form key1=value1:key2=walue2... **/
	sprintf(option_str ,sizeof(option_str) ,"sample_fmt=&s:sample_rates=%d:channel_layout=0x%"PRIx64 ,av_get_sample_fmt_name(AV_SAMPLE_FMT_S16) ,44100 ,(uint64_t)AV_CH_LAYOUT_STEREO);
	err = avfilter_init_str(aform);


}

