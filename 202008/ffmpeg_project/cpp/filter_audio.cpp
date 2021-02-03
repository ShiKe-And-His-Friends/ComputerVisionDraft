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

static int init_filter(const char* filter_detscr) {

	int err;
	char option_str[2048];
	const AVFilter* abuffersrc = avfilter_get_by_name("abuffer");
	const AVFilter* abuffersink = avfilter_get_by_name("abuffersink");
	AVFilterInOut* outputs = avfilter_inout_alloc();
	AVFilterInOut* inputs = avfilter_inout_alloc();
	static const enum AVSampleFormat out_sample_fmts[] = { AV_SAMPLE_FMT_S16 ,AV_SAMPLE_FMT_NONE };
	static const int64_t out_channel_layouts[] = { AV_CH_LAYOUT_MONO ,-1 };
	static const int out_sample_rates[] = { 8000 ,-1 };
	const AVFilterLink* outLink;
	AVRational time_base = aformatCtx->streams[audio_stream_index]->time_base;

	filterGrap = avfilter_graph_alloc();
	if (!inputs || !outputs || !filterGrap) {
		av_log(nullptr, AV_LOG_ERROR, "filter context alloc failure.\n");
		err = AVERROR(ENOMEM);
		goto end;
	}
	if (!acodecCtx->channel_layout) {
		acodecCtx->channel_layout = av_get_default_channel_layout(acodecCtx->channels);
	}
	snprintf(option_str ,sizeof(option_str) ,"time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%" PRIx64 ,time_base.num ,time_base.den ,acodecCtx->sample_rate ,av_get_sample_fmt_name(acodecCtx->sample_fmt) ,acodecCtx->channel_layout);
	err = avfilter_graph_create_filter(&abufferFilterCtx ,abuffersrc ,"in" ,option_str ,nullptr , filterGrap);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph abuffer src context failure.\n");
		goto end;
	}
	err = avfilter_graph_create_filter(&sinkFilterCtx, abuffersink, "out", nullptr, nullptr, filterGrap);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph abuufer sink context failure.\n");
		goto end;
	}
	err = av_opt_set_int_list(sinkFilterCtx ,"sample_fmts" , out_sample_fmts ,-1 , AV_OPT_SEARCH_CHILDREN);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph sample fmts failure.\n");
		goto end;
	}
	err = av_opt_set_int_list(sinkFilterCtx, "channel_layouts", out_channel_layouts, -1, AV_OPT_SEARCH_CHILDREN);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph channel_layouts failure.\n");
		goto end;
	}
	err = av_opt_set_int_list(sinkFilterCtx, "sample_rates", out_sample_rates, -1, AV_OPT_SEARCH_CHILDREN);
	if (err < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph sample fmts failure.\n");
		goto end;
	}
	
	outputs->name = av_strdup("in");
	outputs->filter_ctx = abufferFilterCtx;
	outputs->pad_idx = 0;
	outputs->next = nullptr;

	inputs->name = av_strdup("out");
	inputs->filter_ctx = sinkFilterCtx;
	inputs->pad_idx = 0;
	inputs->next = nullptr;

	if ((err = avfilter_graph_parse_ptr(filterGrap ,filter_detscr ,&inputs ,&outputs ,nullptr)) < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph parse failure.\n");
		goto end;
	}
	if ((err = avfilter_graph_config(filterGrap , nullptr)) < 0) {
		av_log(nullptr, AV_LOG_ERROR, "filter graph config failure.\n");
		goto end;
	}
	outLink = sinkFilterCtx->inputs[0];
	av_get_channel_layout_string(option_str ,sizeof(option_str) ,-1 ,outLink->channel_layout);
	av_log(NULL, AV_LOG_INFO, "Output: srate:%dHz chlayout:%s\n",
		(int)outLink->sample_rate,
		option_str);
end:
	avfilter_inout_free(&inputs);
	avfilter_inout_free(&outputs);
	return err;
}

static void print_frame(const AVFrame* frame)
{
	const int n = frame->nb_samples * av_get_channel_layout_nb_channels(frame->channel_layout);
	const uint16_t* p = (uint16_t*)frame->data[0];
	const uint16_t* p_end = p + n;
	while (p < p_end) {
		fputc(*p, stdout);
		fputc(*p >> 16, stdout);
		p++;
	}
	fflush(stdout);
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
	ret = av_find_best_stream(aformatCtx ,AVMEDIA_TYPE_AUDIO ,-1 ,-1 ,&codec ,0);
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
	av_log(nullptr, AV_LOG_INFO, "open file success.\n");
	return 0;
}

int main(int argc ,char** argv) {
	int ret;
	AVPacket packet;
	AVFrame* frame = av_frame_alloc();
	AVFrame* filter_frame = av_frame_alloc();
	const static char* filter_desrc = "aresample=8000,aformat=sample_fmts=s16:channel_layouts=mono";
	const static char* player = "ffplay -f s16le -ar 8000 -ac 1 -";

	if (!frame || !filter_frame) {
		fprintf(stderr, "frame alloc failure.\n");
		return -1;
	}

	if (argc !=2) {
		fprintf(stderr ,"Usage : %s file | %s \n" ,argv[0] , player);
		return -1;
	}

	if ((ret = open_input_file(argv[1])) < 0) {
		goto end;
	}
	if ((ret = init_filter(filter_desrc) < 0)) {
		goto end;
	}
	// read all packet
	while (1) {
		if ((ret = av_read_frame(aformatCtx ,&packet)) < 0) {
			break;
		}
		if (packet.stream_index == audio_stream_index) {
			ret = avcodec_send_packet(acodecCtx ,&packet);
			if (ret < 0) {
				av_log(nullptr, AV_LOG_ERROR, "send packet failure. %d  %d\n", ret, audio_stream_index);
				break;
			}
			while (ret >= 0) {
				ret = avcodec_receive_frame(acodecCtx ,frame);
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				} else if (ret < 0) {
					av_log(nullptr, AV_LOG_ERROR, "receive frame failure.\n");
					goto end;
				}
				if (ret >= 0) {
					if (av_buffersrc_add_frame_flags(abufferFilterCtx ,frame ,AV_BUFFERSRC_FLAG_KEEP_REF)< 0) {
						av_log(nullptr, AV_LOG_ERROR, "filter src add frame failure.\n");
						break;
					}
					while (1) {
						ret = av_buffersink_get_frame(sinkFilterCtx, filter_frame);
						if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
							break;
						}
						if (ret < 0) {
							goto end;
						}
						print_frame(filter_frame);
						av_frame_unref(filter_frame);
					}
					av_frame_unref(frame);
				}

			}
		}
		av_packet_unref(&packet);
	}

	return 0;

end:
	avfilter_graph_free(&filterGrap);
	avcodec_free_context(&acodecCtx);
	avformat_close_input(&aformatCtx);
	av_frame_free(&frame);
	av_frame_free(&filter_frame);
	if (ret < 0 && ret != AVERROR_EOF) {
		char info[256] = { 0 };
		av_strerror(ret, info, sizeof(info));
		av_log(nullptr, AV_LOG_ERROR, "Error occurred %s \n", info);
	}
	return -1;
}