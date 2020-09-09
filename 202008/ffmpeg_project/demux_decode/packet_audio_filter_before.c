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

	AVDictionary *options_disc = NULL;
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


}

