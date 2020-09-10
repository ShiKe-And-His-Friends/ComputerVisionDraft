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
	err = avfilter_init_str(aformat_ctx ,option_str);
	if (err < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Could not initialize the aformat filter.\n");
		return err;
	}
	/** Finally create the abuffersink filter; It will be used to get the filtered data out of the graph. **/
	abuffersink = avfilter_get_by_name("abuffersink");
	if (!abuffersink) {
		fprintf(stderr ,"Could not find the abuffersink filter.\n");
		return AVERROR_FILTER_NOT_FOUND;
	}
	abuffersink_ctx = avfilter_graph_alloc_filter(filter_graph ,abuffersink ,"sink");
	if (!abuffersink_ctx) {
		fprintf(stderr ,"Could not allocate the abuffersink instance.\n");
		return AVERROR(ENOMEM);
	}
	/** This filter takes no options. **/
	err = avfilter_init_str(abuffersink_ctx ,NULL);
	if (err < 0) {
		fprintf(stderr ,"Could not initialize the abuffersink instance.\n");
		return err;
	}
	/** Connect the filters in this simple case the filters just from a linear chain. **/
	err = avfilter_link(abuffer_ctx ,0 ,volume_ctx ,0);
	if (err >= 0) {
		err = avfilter_link(volume_ctx ,0 ,aformat_ctx ,0);
	}
	if (err >= 0) {
		err = avfilter_link(aformat_ctx ,0 ,abuffersink_ctx ,0);
	}
	if (err < 0) {
		fprintf(stderr ,"Error connecting filters\n");
		return err;
	}
	/** Configure the graph. **/
	err = avfilter_graph_config(filter_graph ,NULL);
	if (err < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Error configuring the filter graph.\n");
		return err;
	}

	*graph = filter_graph;
	*src = abuffer_ctx;
	*sink = abuffersink_ctx;

	return 0;
}


/** Do something uesful with the filtered data: this sample example ust prints the MD5 checksum of each plane to stdout. **/

static int process_output(struct AVMD5 *md5 ,AVFrame *frame) {
	int planar = av_sample_fmt_is_planner(frame->format);
	int channels = av_get_channel_layout_nb_channels(frame->channel_layout);
	int planes = planar ? channels : 1;
	int bps = av_get_bytes_per_sample(frame->format);
	int plane_size = bps * frame->nb_samples * (planar? 1 : channels);
	int i,j;

	for(i = 0 ; i < plannes ; i++) {
		uint8_t checksum[16];
		av_md5_init(md5);
		av_md5_sum(checksum ,frame->extended_data[i] ,plane_size);

		fprintf(stdout ,"plane %d: 0x" ,i);
		for (j = 0 ; j < sizeof(checksum) ; j++) {
			fprintf(stdout ,"%02X" ,checksum[j]);
		}
		fprintf(stdout ,"\n");
	}
	fprintf(stdout ,"\n");

	return 0;
}

/** Construct a frame of audio data to be filtered; This sample example just synthesizes a since wave. **/
static int get_input(AVFrame *frame ,int frame_num) {
	int err ,i ,j;
#define FRAME_SIZE 1024
	/** Set up the frame properties and allocate the bufferfor the data. **/
	frame->sample_rate = INPUT_SAMPLERATE;
	frame->format = INPUT_FORMAT;
	frame->channel_layout = INPUT_CHANNEL_LAYOUT;
	frame->nb_samples = FRAME_SIZE;
	frame->pts = frame_num * FRAME_SIZE;

	err = av_frame_get_buffer(frame ,0);
	if (err < 0) {
		return err;
	}

	/** Fill the data for each channel. **/
	for (i = 0 ; i < 5 ; i ++) {
		float *data = (float *)frame->extended_data[i];
		for (j = 0 ; j < frame->nb_samples ; j++) {
			data[j]  = sin(2 * M_PI * (frame_num + j) * (i+1) / FRAME_SIZE);
		}
	}

	return 0;
}

int main(int argc ,char *argv[]) {
	struct AVMD5 *md5;
	AVFilterGraph *graph;
	AVFilterContext *src ,*sink;
	AVFrame *frame;
	uint8_t errstr[1024];
	float duration;
	int err ,nb_frames ,i;
	if (argc < 2) {
		fprintf(stderr ,"Usage : %s <duration> \n" ,argv[0]);
		return 1;
	}
	duration = atof(argv[1]);
	nb_frames = duration * INPUT_SAMPLERATE / FRAME_SIZE;
	if (nb_frames < 0) {
		fprintf(stderr ,"Invalid duration %s \n" ,argv[1]);
		return 1;
	}
	/** Allocate the frame we will be using to store the data **/
	frame = av_frame_alloc();
	if (!frame) {
		fprintf(stderr ,"Error allocating the frame.\n");
		return 1;
	}
	md5 = av_md5_alloc();
	if (!md5) {
		fprintf(stderr ,"Error allocating the MD5 context.\n");
		return 1;
	}
	/** Set up the filtergraph **/
	err = init_filter_graph(&graph ,&src ,&sink);
	if (err < 0) {
		fprintf(stderr ,"Unable to init filter graph:");
		goto fail;
	}

	/** The main filtering loop **/
	for (i = 0 ; i < nb_frames ; i++) {
		/** get an input frame to be filtered **/
		err = get_input(frame ,i);
		if (err < 0) {
			fprintf(stderr ,"Error generating input frame:");
			goto fail;
		}
		/** send the frame to the input of the filtergraph. **/
		err = av_buffersrc_add_frame(src ,frame);
		if (err < 0) {
			av_frame_unref(frame);
			fprintf(stderr ,"Error submitting the frame to filtergraph:");
		}
		/** get all the filtered output that is available. **/
		while ((err = av_buffersink_get_frame(sink ,frame)) >= 0) {
			/** now do something with our filtered frame **/
			err = process_output(md5 ,frame);
			if (err < 0) {
				fprintf(stderr ,"Error processing the filtered frame:");
				goto fail;
			}
			av_frame_unref(frame);
		}
		if (err == AVERROR(EAGAIN)) {
			/** need to feed more frames in. **/
			continue;
		} else if (err == AVERROR_EOF) {
			/** nothing more to do ,finish. **/
			break;
		} else if (err < 0) {
			/** An error occurred. **/
			fprintf(stderr ,"Error filtering the data:");
			goto fail;
		}
	}

	av_filter_graph_free(&graph);
	av_frame_free(&frame);
	av_free(&md5);
	return 0;

fail:
	av_strerror(err ,errstr ,sizeof(errstr));
	fprintf(stderr ,"%s\n" ,errstr);
	return 1;
}
