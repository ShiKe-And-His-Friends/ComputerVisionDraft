#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>

typedef struct FilteringContext{
	AVFilterContext* buffersink_ctx;
	AVFilterContext* buffersrc_ctx;
	AVFilterGraph* filter_graph;
} FilteringContext;

typedef struct StreamContext {
	AVCodecContext* dec_ctx;
	AVCodecContext* enc_ctx;
}StreamContext;

int interrupt_cb(void* ctx) {
	// fprintf(stderr ,"interrupt cb.\n");
	return 0;
}

static int init_filter(FilteringContext* fctx ,AVCodecContext* dec_ctx 
		,AVCodecContext* enc_ctx ,char* space) {
	char args[512];
	int ret = 0;
	const AVFilter* buffersrc = NULL;
	const AVFilter* buffersink = NULL;
	AVFilterContext* buffersrc_ctx = NULL;
	AVFilterContext* buffersink_ctx = NULL;
	AVFilterInOut* outputs = avfilter_inout_alloc();
	AVFilterInOut* inputs = avfilter_inout_alloc();
	AVFilterGraph* filter_graph = avfilter_graph_alloc();
	if (!outputs || !inputs || !filter_graph) {
		ret = AVERROR(ENOMEM);
		goto end;
	}
	if (dec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
		buffersrc = avfilter_get_by_name("buffer");
		buffersink = avfilter_get_by_name("buffersink");
		if (!buffersrc || ! buffersink) {
			av_log(NULL ,AV_LOG_ERROR ,"filter source or sink element not found\n");
			ret = AVERROR_UNKNOWN;
			goto end;
		}
		snprintf(args ,sizeof(args) ,
			"video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
			dec_ctx->width ,dec_ctx->height ,dec_ctx->pix_fmt
			,dec_ctx->time_base.num ,dec_ctx->time_base.den
			,dec_ctx->sample_aspect_ratio.num
			,dec_ctx->sample_aspect_ratio.den);
		ret = avfilter_graph_create_filter(&buffersrc_ctx ,buffersrc ,"in" ,args ,NULL ,filter_graph);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Can not create buffer source\n");
			goto end;
		}
		ret = avfilter_graph_create_filter(&buffersink_ctx ,buffersink ,"out" ,NULL ,NULL ,filter_graph);

		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Can not create buffer sink\n");
			goto end;
		}
		ret = av_opt_set_bin(buffersink_ctx ,"pix_fmts" ,(uint8_t*)&enc_ctx->pix_fmt ,sizeof(enc_ctx->pix_fmt) ,AV_OPT_SEARCH_CHILDREN);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Can not set output pixel format\n");
			goto end;
		}
	} else if (dec_ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
	
		buffersrc = avfilter_get_by_name("abuffer");
		buffersink = avfilter_get_by_name("abuffersink");
		if (!buffersrc || ! buffersink) {
			av_log(NULL ,AV_LOG_ERROR ,"filter source or sink element not found\n");
			ret = AVERROR_UNKNOWN;
			goto end;
		}
		if (!dec_ctx->channel_layout) {
			dec_ctx->channel_layout = av_get_default_channel_layout(dec_ctx->channels);
		}
		snprintf(args ,sizeof(args) ,
			"time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%"PRIx64 ,
			dec_ctx->time_base.num ,dec_ctx->time_base.den
			,dec_ctx->sample_rate ,av_get_sample_fmt_name(dec_ctx->sample_fmt)
			,dec_ctx->channel_layout);
		ret = avfilter_graph_create_filter(&buffersrc_ctx ,buffersrc ,"in" ,args ,NULL ,filter_graph);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"can not create audio buffer source\n");
			goto end;
		}
		ret = avfilter_graph_create_filter(&buffersink_ctx ,buffersink ,"out" ,NULL ,NULL ,filter_graph);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Can not create audio buffer sink\n");
			goto end;
		}
		ret = av_opt_set_bin(buffersink_ctx ,"channel_layouts" 
				,(uint8_t*)&enc_ctx->channel_layout
				,sizeof(enc_ctx->channel_layout)
				,AV_OPT_SEARCH_CHILDREN);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Cannot set output channel layout\n");
			goto end;
		}
		ret = av_opt_set_bin(buffersink_ctx ,"sample_rates" 
				,(uint8_t*)&enc_ctx->sample_rate 
				,sizeof(enc_ctx->sample_rate)
				,AV_OPT_SEARCH_CHILDREN);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Can not set output sample rate\n");
			goto end;
		}
	} else {
		ret = AVERROR_UNKNOWN;
		goto end;
	}
	outputs->name = av_strdup("in");
	outputs->filter_ctx = buffersrc_ctx;
	outputs->pad_idx = 0;
	outputs->next = NULL;

	inputs->name = av_strdup("out");
	inputs->filter_ctx = buffersink_ctx;
	inputs->pad_idx = 0;
	inputs->next = NULL;
	if (!outputs->name || !inputs->name) {
		ret = AVERROR(ENOMEM);
		goto end;
	}
	if ((ret = avfilter_graph_parse_ptr(filter_graph ,space 
		,&inputs ,&outputs ,NULL)) < 0) {
		goto end;
	}
	if ((ret = avfilter_graph_config(filter_graph ,NULL) < 0)) {
		goto end;
	}
	
	/* Fill FilteringContext */
	fctx->buffersrc_ctx = buffersrc_ctx;
	fctx->buffersink_ctx = buffersink_ctx;
	fctx->filter_graph = filter_graph;
end:
	avfilter_inout_free(&inputs);
	avfilter_inout_free(&outputs);
	return ret;
}

int main(int argc ,char* argv[]) {
	int ret;
	char* output = NULL;
	char* input = NULL;
	AVFormatContext* context = NULL;
	AVFormatContext* output_context = NULL;
	AVDictionary* format_opts = NULL;
	AVCodec* codec = NULL;
	AVStream* in_stream = NULL;
	AVStream* out_stream = NULL;
	StreamContext* stream_ctx = NULL;

	if (argc != 3) {
		fprintf(stderr ,"input format error.\n %s <input_file> <output_file> \n" ,argv[0]);
		return -1;
	}

	input = argv[1];
	output = argv[2];

	// open input file	
	context = avformat_alloc_context();
	context->interrupt_callback.callback = interrupt_cb;
	ret = avformat_open_input(&context ,input ,NULL ,&format_opts);
	if (!ret) {
		av_dump_format(context ,0 ,input ,0);
	} else {
		return 0;
	}
	stream_ctx = av_mallocz_array(context->nb_streams ,sizeof(*stream_ctx));
	if (!stream_ctx) {
		return AVERROR(ENOMEM);
	}
	ret = avformat_find_stream_info(context ,NULL);
	if (ret) {
		fprintf(stderr ,"Find stream best info failure.\n");
		return 0;
	}
	
	for (int i = 0 ; i < context->nb_streams ;i++) {
		AVStream* stream = context->streams[i];
		AVCodec* dec = avcodec_find_decoder(stream->codecpar->codec_id);
		AVCodecContext* codec_ctx = NULL;
		if (!dec) {
			av_log(NULL ,AV_LOG_ERROR ,"Fialure to find decoder for stream#%u\n" ,i);
			return AVERROR_DECODER_NOT_FOUND;
		}
		codec_ctx = avcodec_alloc_context3(dec);
		
		if (!codec_ctx) {
			av_log(NULL ,AV_LOG_ERROR ,"Fialure to allocate the decoder context \n");
			return AVERROR(ENOMEM);
		}
		ret = avcodec_parameters_to_context(codec_ctx ,stream->codecpar);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Failure to copy decoder paramters to input decoder codec for stream#%u" ,i);
			return ret;
		}
		av_log(NULL ,AV_LOG_ERROR ,"opening stream#%u\n" ,i);
		if (codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO
				|| codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
			if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
				codec_ctx->framerate = av_guess_frame_rate(context ,stream ,NULL);
				codec_ctx->time_base = av_inv_q(codec_ctx->framerate);
				av_log(NULL ,AV_LOG_ERROR ,"MEDIA VIDEO\n");
			}
			if (codec_ctx->codec_type == AVMEDIA_TYPE_AUDIO) {
				codec_ctx->time_base = (AVRational){1 ,codec_ctx->sample_rate};
				av_log(NULL ,AV_LOG_ERROR ,"MEDIA AUDIO\n");
			}
			ret = avcodec_open2(codec_ctx ,dec ,NULL);
			if (ret < 0) {
				av_log(NULL ,AV_LOG_ERROR ,"Failed to open decoder for stream#%u\n" ,i);
				return ret;
			} else {
				
				av_log(NULL ,AV_LOG_ERROR ,"Success to open decoder for stream#%u\n" ,i);
			}
		}
		stream_ctx[i].dec_ctx = codec_ctx;
	}

	fprintf(stderr ,"Open Success\n");

	// open output file
	AVCodec* h264Codec = NULL;
	avformat_alloc_output_context2(&output_context ,NULL ,NULL ,output);	
	AVCodecContext* decode_ctx = NULL;
	AVCodecContext* encode_ctx = NULL;
	if (!output_context) {
		av_log(NULL ,AV_LOG_ERROR ,"Could not open output context.\n");
		return AVERROR_UNKNOWN;
	}
	for (int i = 0 ; i < context->nb_streams ; i++) {
		out_stream = avformat_new_stream(output_context ,NULL);
		if (!out_stream) {
			av_log(NULL ,AV_LOG_ERROR ,"Failed allocting output stream.\n");
			return AVERROR_UNKNOWN;
		}
		in_stream = context->streams[i];
		decode_ctx = stream_ctx[i].dec_ctx;
		if (decode_ctx->codec_type == AVMEDIA_TYPE_AUDIO
				|| decode_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
			// sudo vim /etc/ld.so.conf #libx264 update
			h264Codec = avcodec_find_encoder(decode_ctx->codec_id);
			if (!h264Codec) {
				av_log(NULL ,AV_LOG_ERROR ,"Necessary encoder not found.\n");
				return AVERROR_INVALIDDATA;
			}
			encode_ctx = avcodec_alloc_context3(h264Codec);
			if (!encode_ctx) {
				av_log(NULL ,AV_LOG_ERROR ,"Necessary encoder context not found.\n");
				return AVERROR_INVALIDDATA;
			}

			if (decode_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {
				encode_ctx->height = decode_ctx->height;
				encode_ctx->width = decode_ctx->width;
				encode_ctx->sample_aspect_ratio = decode_ctx->sample_aspect_ratio;
				if (h264Codec->pix_fmts) {
					encode_ctx->pix_fmt = h264Codec->pix_fmts[0];
				} else {
					encode_ctx->pix_fmt = decode_ctx->pix_fmt;
				}
				encode_ctx->time_base = av_inv_q(decode_ctx->framerate);
				printf("\nVIDEO\n");
			} else {
				encode_ctx->sample_rate = decode_ctx->sample_rate;
				encode_ctx->channel_layout = decode_ctx->channel_layout;
				encode_ctx->channels = av_get_channel_layout_nb_channels(encode_ctx->channel_layout);
				encode_ctx->sample_fmt = h264Codec->sample_fmts[0];
				encode_ctx->time_base = (AVRational) {1 ,encode_ctx->sample_rate};
				printf("\nAUDIO\n");
			}
			if (output_context->oformat->flags & AVFMT_GLOBALHEADER) {
				encode_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
			}


			AVDictionary* options = NULL;
			// encode_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
			ret = avcodec_open2(encode_ctx ,h264Codec ,&options);
			if (ret < 0) {
				fprintf(stderr ,"Open Out Codec Failure.\n");
				return -1;
			} else {
				fprintf(stderr ,"Open Out Success stream#%u\n" ,i);
			}
			ret = avcodec_parameters_from_context(out_stream->codecpar ,encode_ctx);
			if (ret < 0) {
				av_log(NULL ,AV_LOG_ERROR ,"Failed to copy encoder parmeters stream#%u" ,i);
				return ret;
			}
			out_stream->time_base = encode_ctx->time_base;
			stream_ctx[i].enc_ctx = encode_ctx;
		} else if (decode_ctx->codec_type == AVMEDIA_TYPE_UNKNOWN) {
			av_log(NULL ,AV_LOG_FATAL ,"Elementary stream#%d is of unknown type, cannot proceed\n" ,i);
			return AVERROR_INVALIDDATA;
		} else {
			/* MUST REMUXED */
			ret = avcodec_parameters_copy(out_stream->codecpar ,in_stream->codecpar);
			if (ret < 0) {
				av_log(NULL ,AV_LOG_ERROR ,"Copying parameters for stream#%u failed.\n" ,i);
				return ret;
			}
			out_stream->time_base = in_stream->time_base;
		}
	}
	av_dump_format(output_context ,0 ,output ,1);

	//init filter
	static FilteringContext* filterCtx = NULL;
	char* space;
	filterCtx = av_malloc_array(context->nb_streams ,sizeof(*filterCtx));
	if (!filterCtx) {
		return AVERROR(ENOMEM);
	}
	for (int i = 0 ; i < context->nb_streams ;i++) {
		filterCtx[i].buffersrc_ctx = NULL;
		filterCtx[i].buffersink_ctx = NULL;
		filterCtx[i].filter_graph = NULL;
		if (!(context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO || context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) ) {
			continue;
		}

		if (context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
			space = "null";
		} else {
			space = "anull";
		}
		ret = init_filter(&filterCtx[i] ,stream_ctx[i].dec_ctx 
				,stream_ctx[i].enc_ctx ,space);
		if (ret < 0) {
			goto end;
		}
	}

	// read packet
	int stream_index = -1;
	enum AVMediaType type;
	AVPacket packet;
	AVFrame* frame;
	AVFrame* filte_frame;
	int got_frame;
	int (*dec_func)(AVCodecContext* ,AVFrame* ,int* ,const AVPacket*);
	while(1) {
		if ((ret = av_read_frame(context ,&packet)) < 0) {
			av_log(NULL ,AV_LOG_INFO ,"read packect from file result end.\n");
			break;
		}
		stream_index = packet.stream_index;
		type = context->streams[stream_index]->codecpar->codec_type;
		av_log(NULL ,AV_LOG_DEBUG ,"Demuxer gave frame of stream index %u\n" ,stream_index);
		if (filterCtx[stream_index].filter_graph) {
			av_log(NULL ,AV_LOG_DEBUG ,"Going recodec&filter the frame\n");
			frame = av_frame_alloc();
			if (!frame) {
				ret = AVERROR(ENOMEM);
				av_log(NULL ,AV_LOG_ERROR ,"GET FRAME FAILURE\n");
				break;
			}
			av_packet_rescale_ts(&packet ,context->streams[stream_index]->time_base ,stream_ctx[stream_index].dec_ctx->time_base);
			dec_func = (type == AVMEDIA_TYPE_VIDEO) ? avcodec_decode_video2 : avcodec_decode_audio4;
			ret = dec_func(stream_ctx[stream_index].dec_ctx ,frame ,&got_frame ,&packet);
			if (ret < 0) {
				av_frame_free(&frame);
				av_log(NULL ,AV_LOG_ERROR ,"Deocding failed.\n");
			}
			if (got_frame) {
				frame->pts = frame->best_effort_timestamp;
				//TODO add write codec frame
				av_log(NULL ,AV_LOG_DEBUG ,"Pushing decoded frame to filters.\n");
				ret = av_buffersrc_add_frame_flags(filterCtx[stream_index].buffersrc_ctx ,frame ,0);
				if (ret < 0) {
					av_log(NULL ,AV_LOG_ERROR ,"Error while feeding the filtergraph.\n");
					ret = AVERROR_UNKNOWN;
					goto end;
				}
				while(1) {
					filte_frame = av_frame_alloc();
					if(!filte_frame) {
						av_log(NULL ,AV_LOG_ERROR ,"filter frame alloc failure.\n");
						ret = AVERROR(ENOMEM);
						goto end;
					}
					av_log(NULL ,AV_LOG_DEBUG ,"Pulling filtered frame from filters.\n");
					ret = av_buffersink_get_frame(filterCtx[stream_index].buffersink_ctx ,filte_frame);
					if (ret < 0) {
						/**
						 * If no more frame into sink filter - returen AVERROR(EAGAIN)
						 * If no more frame for output - return AVERROR_EOF;
						 * rewrite and retcode to 0 to show it as normal procedure completion.
						 * */
						if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
							ret = 0;
						}
						av_frame_free(&filte_frame);
						break;
					}
					filte_frame->pict_type = AV_PICTURE_TYPE_NONE;
					int got_frame_local;
					AVPacket enc_pkt;
					int (*enc_func)(AVCodecContext* ,AVPacket* ,const AVFrame* ,int*) = 
						(context->streams[stream_index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) ? avcodec_encode_video2 : avcodec_encode_audio2;
					if (!got_frame) {
						av_log(NULL ,AV_LOG_ERROR ,"got frame failure\n");
						got_frame = &got_frame_local;
					}
					av_log(NULL ,AV_LOG_DEBUG ,"Encodeing frame.\n");
					enc_pkt.data = NULL;
					enc_pkt.size = 0;
					av_init_packet(&enc_pkt);
					ret = enc_func(stream_ctx[stream_index].enc_ctx ,&enc_pkt ,filte_frame ,&got_frame);
					av_frame_free(&filte_frame);
					if (ret < 0) {
						break;
					}
					if (!(&got_frame)) {
						ret = 0;
						av_log(NULL ,AV_LOG_ERROR ,"Not info got frame.\n");
						break;	
					}
					enc_pkt.stream_index = stream_index;
					av_packet_rescale_ts(&enc_pkt ,stream_ctx[stream_index].enc_ctx->time_base ,output_context->streams[stream_index]->time_base);
					ret = av_interleaved_write_frame(output_context ,&enc_pkt);

				}
				if (ret < 0) {
					av_log(NULL ,AV_LOG_ERROR ,"recodec rescale failure in sink filter\n");
					goto end;
				}
				av_frame_free(&frame);
			} else {
				av_frame_free(&frame);
			}
		} else {
			/** rescale frame without recode */
			av_log(NULL ,AV_LOG_INFO ,"rescale frame without recodeing.\n");
			av_packet_rescale_ts(&packet ,context->streams[stream_index]->time_base ,output_context->streams[stream_index]->time_base);
			ret = av_interleaved_write_frame(output_context ,&packet);
			if (ret < 0) {
				goto end;
			}
		}
		av_packet_unref(&packet);	
	}

	fprintf(stderr ,"\nDRAW TEXT FINISH\n");
	return 0;

end:
	fprintf(stderr ,"\nDRAW TEXT ERROR\n");
	return -1;
}
