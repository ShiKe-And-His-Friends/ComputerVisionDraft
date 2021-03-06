extern "C" {
	#include <libavformat/avformat.h>
	#include <libavcodec/avcodec.h>
	#include <libavfilter/avfilter.h>
	#include <libavfilter/buffersrc.h>
	#include <libavfilter/buffersink.h>
	#include <libavutil/channel_layout.h>	
	#include <libavutil/samplefmt.h>
	#include <libavutil/opt.h>
	#include <libavutil/md5.h>
}
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

#define INPUT_SAMPLERATE 48000
#define INPUT_FORMAT AV_SAMPLE_FMT_FLTP
#define INPUT_CHANNEL_LAYOUT AV_CH_LAYOUT_5POINT0
#define VOLUME_VAL 0.9
#define FRAME_SIZE 2048

using namespace std;

static int init_filter_graph(AVFilterGraph** graphInf ,AVFilterContext** bufferInf , AVFilterContext** sinkInf) {

	AVFilterGraph* filter_graph;
	const AVFilter* abuffer;
	AVFilterContext* abufferContext;
	const AVFilter* volume;
	AVFilterContext* volumeContext;
	AVDictionary* dictionary = nullptr;
	const AVFilter* format;
	AVFilterContext* formatContext;
	const AVFilter* abuffersink;
	AVFilterContext* abuffersinkContext;

	int err;
	char ch_layout[64];
	char option_str[2048];

	filter_graph = avfilter_graph_alloc();
	if (filter_graph == nullptr || !filter_graph) {
		cout << "init abuffer filter graph failure." << endl;
		return AVERROR(ENOMEM);
	}

	abuffer = avfilter_get_by_name("abuffer");
	if (abuffer == nullptr || !abuffer) {
		cout << "init abuffer filter failure." << endl;
		return AVERROR(ENOMEM);
	}

	abufferContext = avfilter_graph_alloc_filter(filter_graph ,abuffer ,"src");
	if (abufferContext == nullptr || !abufferContext) {
		cout << "init abuffer filter context failure." << endl;
		return AVERROR(ENOMEM);
	}

	// set filter option through AVOption API
	av_get_channel_layout_string(ch_layout , sizeof(ch_layout) ,0 , INPUT_CHANNEL_LAYOUT);
	av_opt_set(abufferContext ,"channel_layout" ,ch_layout ,AV_OPT_SEARCH_CHILDREN);
	av_opt_set(abufferContext, "sample_fmt",av_get_sample_fmt_name(INPUT_FORMAT) , AV_OPT_SEARCH_CHILDREN);
	av_opt_set_q(abufferContext, "time_base", AVRational{1 ,INPUT_SAMPLERATE}, AV_OPT_SEARCH_CHILDREN);
	av_opt_set_int(abufferContext ,"sample_rate" ,INPUT_SAMPLERATE , AV_OPT_SEARCH_CHILDREN);

	err =  avfilter_init_str(abufferContext ,NULL);
	if (err < 0) {
		cout << "src filter config failure." << endl;
		return err;
	}

	// different way set this
	volume = avfilter_get_by_name("volume");
	if (volume == nullptr || !volume) {
		cout << "init volume filter failure." << endl;
		return AVERROR(ENOMEM);
	}
	volumeContext = avfilter_graph_alloc_filter(filter_graph, volume, "volume");
	if (volumeContext == nullptr || !volumeContext) {
		cout << "init volume filter context failure." << endl;
		return AVERROR(ENOMEM);
	}
	av_opt_set(volumeContext, "volume", AV_STRINGIFY(VOLUME_VAL), AV_OPT_SEARCH_CHILDREN);

	/*
		av_dict_set(&dictionary, "volume", AV_STRINGIFY(VOLUME_VAL), 0);
		err = avfilter_init_dict(volumeContext , &dictionary);
		av_dict_free(&dictionary);
	*/
	err = avfilter_init_str(volumeContext, NULL);
	if (err < 0) {
		cout << "volume filter config failure." << endl;
		return err;
	}

	// set avformat
	format = avfilter_get_by_name("aformat");
	if (format == nullptr || !format) {
		cout << "init format filter failure." << endl;
		return AVERROR(ENOMEM);
	}
	formatContext = avfilter_graph_alloc_filter(filter_graph, format, "aformat");
	if (formatContext == nullptr || !formatContext) {
		cout << "init format filter context failure." << endl;
		return AVERROR(ENOMEM);
	}

	////third way set option
	snprintf(option_str ,sizeof(option_str) , "sample_fmts=%s : sample_rates=%d : channel_layouts=0x%" PRIx64,
		av_get_sample_fmt_name(AV_SAMPLE_FMT_S16),
		44100,
		(uint64_t)AV_CH_LAYOUT_STEREO);
	err = avfilter_init_str(formatContext ,option_str);
	if (err < 0) {
		cout << "format filter config failure." << endl;
		return err;
	}

	abuffersink = avfilter_get_by_name("abuffersink");
	if (abuffersink == nullptr || !abuffersink) {
		cout << "init abuffer filter failure." << endl;
		return AVERROR(ENOMEM);
	}
	abuffersinkContext = avfilter_graph_alloc_filter(filter_graph, abuffersink, "sink");
	if (abuffersinkContext == nullptr || !abuffersinkContext) {
		cout << "init abuufersink filter context failure." << endl;
		return AVERROR(ENOMEM);
	}

	// this filter set no option
	err = avfilter_init_str(abuffersinkContext, NULL);
	if (err < 0) {
		fprintf(stderr, "Could not initialize the abuffersinksinke instance.\n");
		return err;
	}

	// connect the filters in this chain samples case from a linear chain.
	err = avfilter_link(abufferContext,0 , volumeContext,0);
	if (err >= 0) {
		err = avfilter_link(volumeContext, 0, formatContext, 0);
	}
	if (err >= 0) {
		err = avfilter_link(formatContext, 0, abuffersinkContext, 0);
	}
	if (err < 0) {
		cout << "src filter connect failure." << endl;
		return err;
	}
	err = avfilter_graph_config(filter_graph ,NULL);
	if(err < 0) {
		av_log(NULL, AV_LOG_ERROR, "Error configuring the filter graph.\n");
		return err;
	}
	*graphInf = filter_graph;
	*bufferInf = abufferContext;
	*sinkInf = abuffersinkContext;
	return 0;
}

static int process_output(struct AVMD5* md5 ,AVFrame* frame) {
	int planar = av_sample_fmt_is_planar((AVSampleFormat)frame->format);
	int channels = av_get_channel_layout_nb_channels(frame->channel_layout);
	int planes = planar ? channels : 1;
	int bps = av_get_bytes_per_sample((AVSampleFormat)frame->format);
	int plane_size = bps * frame->nb_samples * (planar ? 1 : channels);
	int i, j;
	for (i = 0; i < planes; i++) {
		uint8_t checksum[16];
		av_md5_init(md5);
		av_md5_sum(checksum, frame->extended_data[i], plane_size);
		fprintf(stdout, "plane %d: 0x", i);
		for (j = 0; j < sizeof(checksum); j++) {
			fprintf(stdout, "%02X", checksum[j]);
		}
		fprintf(stdout, "\n");
	}
	return 0;
}

static int get_input(AVFrame* frame ,int frame_num) {
	int ret;
	int i, j;
	frame->sample_rate = INPUT_SAMPLERATE;
	frame->format = INPUT_FORMAT;
	frame->channel_layout = INPUT_CHANNEL_LAYOUT;
	frame->nb_samples = FRAME_SIZE;
	frame->pts = static_cast<int64_t>(frame_num * FRAME_SIZE);
	ret = av_frame_get_buffer(frame ,0);
	if (ret < 0) {
		return ret;
	}

	for (i = 0; i < 5; i++) {
		float* data = (float*)frame->extended_data[i];
		for (j = 0; j < frame->nb_samples; j++) {
			data[j] = sin(2 * M_PI * (frame_num + j) * (i + 1) / FRAME_SIZE);
			//cout << "values is " << (data[j]) << endl;
		}
	}

	// not alloc continuous arrays
	/*
		uint8_t** data = frame->extended_data;
		for (i = 0; i < 5; i++) {
			for (j = 0; j < frame->nb_samples; j++) {
				(data[i][j]) = (uint8_t)sin(2 * M_PI * (frame_num + j) * (i + 1) / FRAME_SIZE);
				cout << "shikeDebug size is  values result " << (data[i][j]) << endl;
			}
		}
	*/

	return 0;
}

int main_audio_filter (int argc ,char** argv) {
	int duration = 0 ,nb_frame ,i;
	int ret = -1;
	char errstr[1024];
	AVFilterGraph* graph;
	AVFilterContext* src, *sink;
	AVFrame* frame;
	struct AVMD5* md5;

	if (argc != 2) {
		cout << "Please input format :" << endl << endl;
		cout << argv[0] << " <duration>" <<endl;
		return -1;
	}
	duration = atoi(argv[1]);
	cout << "duration is " << duration << endl;

	nb_frame = duration * INPUT_SAMPLERATE / FRAME_SIZE;

	if (nb_frame < 0) {
		cout << "Invalid values." << endl;
		return -1;
	}

	ret = init_filter_graph(&graph ,&src ,&sink);
	
	if (ret < 0) {
		cout << "init filter graph failure." << endl << endl;
		goto fail;
	}

	frame = av_frame_alloc();
	if (!frame) {
		cout << "frame alloctiong failure" << endl;
		return 1;
	}
	md5 = av_md5_alloc();
	if (!md5) {
		cout << "md5 alloctiong failure" << endl;
		return 1;
	}

	for (i = 0; i < nb_frame; i++) {
		ret = get_input(frame, i);
		if (ret < 0) {
			cout << "get input failure." << endl << endl;
			goto fail;
		}
		ret = av_buffersrc_add_frame(src, frame);
		if (ret < 0) {
			cout << "Error submitting the frame to filtring." << endl << endl;
			av_frame_unref(frame);
		}
		while ((ret = av_buffersink_get_frame(sink, frame)) >= 0) {
			ret = process_output(md5, frame);
			if (ret < 0) {
				cout << "Error processing output data." << endl << endl;
				goto fail;
			}
			av_frame_unref(frame);
		}

		if (ret == AVERROR(EAGAIN)) {
			continue;
		}
		else if (ret == AVERROR_EOF) {
			break;
		}
		else if (ret < 0) {
			goto fail;
		}
	}

	cout << "release success. " << av_version_info() << endl;
	avfilter_graph_free(&graph);
	av_frame_free(&frame);

	return 0;

fail:
	av_strerror(ret ,errstr ,sizeof(errstr));
	return 1;
}