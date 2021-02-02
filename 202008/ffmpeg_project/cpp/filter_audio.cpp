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

int audio_stream_index = -1;
static AVFormatContext* aformatCtx;
static AVCodecContext* acodecCtx;
const static AVCodecParameters* acodecParams;
AVFilterContext* abufferFilterCtx;
AVFilterContext* sinkFilterCtx;
AVFilterGraph* filterGrap;

static int init_filter(const AVFrame* filter_detscr) {

	int err;
	char option_str[2048];
	const AVFilter* abuffersrc = avfilter_get_by_name("abuffer");
	const AVFilter* abuffersink = avfilter_get_by_name("abuffersink");
	AVFilterInOut* outputs = avfilter_inout_alloc();
	AVFilterInOut* inputs = avfilter_inout_alloc();
	static const enum AVSampleFormat out_sample_fmts[] = { AV_SAMPLE_FMT_S16 };
	static const int64_t out_channel_layouts[] = { 8000 };
	const AVFilterLink* outlink;
	AVRational time_base = aformatCtx->streams[audio_stream_index]->time_base;

	filterGrap = avfilter_graph_alloc();
	if (!inputs || !outputs || !filterGrap) {
		err = AVERROR(ENOMEM);
		goto end;
	}
	if (!acodecCtx->channel_layout) {
		acodecCtx->channel_layout = av_get_default_channel_layout(acodecCtx->channels);
	}
	snprintf(option_str ,sizeof(option_str) ,"time_base=%d/%d :sample_fmt=%s : channel_layout=0x%" PRIx64 ,time_base.num ,time_base.den ,acodecCtx->sample_rate ,av_get_sample_fmt_name(acodecCtx->sample_fmt) ,acodecCtx->channel_layout);
	err = avfilter_graph_create_filter(&abufferFilterCtx ,abuffersrc ,"in" ,option_str ,nullptr , filterGrap);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph context failure.\n");
		goto end;
	}
	err = av_opt_set_int_list(sinkFilterCtx ,"sample_fmts" , out_sample_fmts ,-1 , AV_OPT_SEARCH_CHILDREN);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph sample fmts failure.\n");
		goto end;
	}
	err = av_opt_set_int_list(sinkFilterCtx, "sample_rate", out_channel_layouts, -1, AV_OPT_SEARCH_CHILDREN);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph sample rate failure.\n");
		goto end;
	}
	outputs->name = av_strdup("in");
	outputs->filter_ctx = abufferFilterCtx;
	outputs->pad_idx = 0;
	outputs->next = nullptr;


end:

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

static int open_input_file(const char* name) {
	int ret;
	AVCodec* codec;
	if ((ret = avformat_open_input(&aformatCtx ,name ,nullptr ,nullptr)) < 0) {
		av_log(nullptr , AV_LOG_ERROR ,"open input file failure.\n");
		return ret;
	}
	if ((ret = avformat_find_stream_info(aformatCtx ,NULL)) < 0) {
		av_log(nullptr, AV_LOG_ERROR, "open input file info failure.\n");
		return ret;
	}
	ret = av_find_best_stream(aformatCtx ,AVMEDIA_TYPE_AUDIO ,-1 ,-1 ,&codec ,-1);
	if (ret < 0) {
		av_log(nullptr, AV_LOG_ERROR, "open input file find best stream info failure.\n");
		return ret;
	}
	audio_stream_index = ret;
	acodecCtx = avcodec_alloc_context3(codec);
	if (!acodecCtx) {
		return AVERROR(ENOMEM);
	}
	avcodec_parameters_to_context(acodecCtx , aformatCtx->streams[audio_stream_index]->codecpar);
	if ((ret = avcodec_open2(acodecCtx , codec ,nullptr))) {
		av_log(nullptr, AV_LOG_ERROR, "open codec failure.\n");
		return ret;
	}
	return 0;
}

int main(int argc ,char** argv) {
	int ret;
	AVFrame* filter_frame = av_frame_alloc();

	const static char* player = "ffplay -f s16le -ar 8000 -ac 1 -";
	if (argc !=2) {
		fprintf(stderr ,"Usage : %s file | %s \n" ,argv[0] , &player);
		return -1;
	}

	if ((ret = open_input_file(argv[1])) < 0) {
		goto end;
	}
	if ((ret = init_filter(&filter_frame) < 0)) {
		goto end;
	}

	return 0;

end:
	return -1;
}