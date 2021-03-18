#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <stdio.h>

typedef struct StreamContext {
	AVCodecContext* dec_ctx;
	AVCodecContext* enc_ctx;
}StreamContext;

int interrupt_cb(void* ctx) {
	return 0;
} 

int main (int argc ,char* argvs[] ) {
	int ret;
	char* input[2];
	char* output;
	AVFormatContext* inputCtx[2];
	AVFormatContext* outputCtx;
	AVCodecContext* inputCodecCtx[2];
	StreamContext* streamCtx = NULL;

	if (argc != 4) {
		fprintf(stderr ,"Input Format Error.\n %s <input_video_file> <input_picture_file> <output_video_file>\n" ,argvs[0]);
		return 0;
	}
	input[0] = argvs[1];
	input[1] = argvs[2];
	output = argvs[3];
	//Open Input File
	inputCtx[0] = avformat_alloc_context();
	inputCtx[0]->interrupt_callback.callback = interrupt_cb;
	AVDictionary* format_opt = NULL;
	ret = avformat_open_input(&inputCtx[0] ,input[0] ,NULL ,&format_opt);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Input File %s Failure\n" ,input[0]);
		goto end;
	}
	streamCtx = av_mallocz_array(inputCtx[0]->nb_streams ,sizeof(streamCtx));
	if (!streamCtx) {
		av_log(NULL ,AV_LOG_ERROR ,"stream contxt copy failure.\n");
		goto end;
	}

	ret = avformat_find_stream_info(inputCtx[0] ,NULL);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Info Input File %s Failure\n" ,input[0]);
		goto end;
	}
	for (int i = 0; i < inputCtx[0]->nb_streams; i++) {
		AVStream* stream = inputCtx[0]->streams[i];
		AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
		if (!codec) {
			av_log(NULL ,AV_LOG_ERROR ,"Input0 find codec stream#%d failure.\n" ,i);
			ret = AVERROR_DECODER_NOT_FOUND;
			goto end;
		}
		inputCodecCtx[0] = avcodec_alloc_context3(codec);
		if (!inputCodecCtx[0]) {
			av_log(NULL, AV_LOG_ERROR, "Find Decoder Context Input0 Stream#%d failure\n", i);
			goto end;
		}
		ret = avcodec_parameters_to_context(inputCodecCtx[0] ,stream->codecpar);
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Copy Output Codec For File %s Failure\n", input[0]);
			goto end;
		}
		if (inputCodecCtx[0]->codec_type == AVMEDIA_TYPE_AUDIO || inputCodecCtx[0]->codec_type == AVMEDIA_TYPE_VIDEO) {
			//TODO set samplerate
			if (inputCodecCtx[0]->codec_type == AVMEDIA_TYPE_VIDEO) {
				inputCodecCtx[0]->framerate = av_guess_frame_rate(inputCtx[0] ,stream ,NULL);
				inputCodecCtx[0]->time_base = av_inv_q(inputCodecCtx[0]->framerate);
			}
			if (inputCodecCtx[0]->codec_type == AVMEDIA_TYPE_AUDIO) {
				inputCodecCtx[0]->time_base = (AVRational){1 ,inputCodecCtx[0]->sample_rate };
			}
			ret = avcodec_open2(inputCodecCtx[0], codec, NULL);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Open Output Codec For File %s Failure\n", input[0]);
				goto end;
			}

		}
		streamCtx[i].dec_ctx = inputCodecCtx[0];
	}
	av_dump_format(inputCtx[0] ,0 ,input[0] ,0);
		
	inputCtx[1] = avformat_alloc_context();
	inputCtx[1]->interrupt_callback.callback = interrupt_cb;
	ret = avformat_open_input(&inputCtx[1] ,input[1] ,NULL ,&format_opt);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Input File %s Failure\n" ,input[1]);
		goto end;
	}
	ret = avformat_find_stream_info(inputCtx[1] ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Info Input File %s Failure\n" ,input[1]);
		goto end;
	}
	av_dump_format(inputCtx[1] ,0 ,input[1] ,0);
	
	
	AVStream* stream = inputCtx[1]->streams[0];
	AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
	if (!codec) {
		av_log(NULL, AV_LOG_ERROR, "Input1 find codec stream#0 failure.\n");
		ret = AVERROR_DECODER_NOT_FOUND;
		goto end;
	}
	inputCodecCtx[1] = avcodec_alloc_context3(codec);
	if (!inputCodecCtx[1]) {
		av_log(NULL, AV_LOG_ERROR, "Find Decoder Context Input1 Stream#0 failure\n");
		goto end;
	}
	ret = avcodec_parameters_to_context(inputCodecCtx[1], stream->codecpar);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Copy Output Codec For File %s Failure\n", input[1]);
		goto end;
	}
	if (inputCodecCtx[1]->codec_type == AVMEDIA_TYPE_VIDEO) {
		inputCodecCtx[1]->framerate = av_guess_frame_rate(inputCtx[0], stream, NULL);
		inputCodecCtx[1]->time_base = av_inv_q(inputCodecCtx[0]->framerate);
	}
	ret = avcodec_open2(inputCodecCtx[1], codec, NULL);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Open Output Codec For File %s Failure\n", input[1]);
		goto end;
	}

	// Open Output File
	ret = avformat_alloc_output_context2(&outputCtx ,NULL ,NULL ,output);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Ouput File %s Open Failure\n" ,output);
		goto end;
	}
	for (int i = 0 ; i < inputCtx[0]->nb_streams ; i++) {		
		AVCodecContext* outputCodecCtx;
		AVCodecContext* inputCodecCtx;
		AVStream* out_stream = avformat_new_stream(outputCtx ,NULL);
		if (!out_stream) {
			av_log(NULL ,AV_LOG_ERROR ,"Ouput File %s Open Stream#%u Failure\n" ,output ,i);
			ret = AVERROR_UNKNOWN;
			goto end;
		}
		AVStream* in_stream = inputCtx[0]->streams[i];
		inputCodecCtx = streamCtx[i].dec_ctx;
		if (inputCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO || inputCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO) {
			AVCodec* codec = avcodec_find_encoder(inputCodecCtx->codec_id);
			if (!codec) {
				av_log(NULL, AV_LOG_ERROR, "Find Codec %s Open Stream#%u Failure\n", output, i);
				goto end;
			}
			outputCodecCtx = avcodec_alloc_context3(codec);
			if (!outputCodecCtx) {
				av_log(NULL, AV_LOG_ERROR, "Find Codec Context %s Open Stream#%u Failure\n", output, i);
				ret = AVERROR(ENOMEM);
				goto end;
			}
			if (inputCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO) {
				outputCodecCtx->width = inputCodecCtx->width;
				outputCodecCtx->height = inputCodecCtx->height;
				outputCodecCtx->sample_aspect_ratio = inputCodecCtx->sample_aspect_ratio;
				if (codec->pix_fmts) {
					outputCodecCtx->pix_fmt = codec->pix_fmts[0];
				}else {
					outputCodecCtx->pix_fmt = inputCodecCtx->pix_fmt;
				}
				outputCodecCtx->framerate = inputCodecCtx->framerate;
				outputCodecCtx->time_base = av_inv_q(inputCodecCtx->framerate);
			} else {
				outputCodecCtx->sample_rate = inputCodecCtx->sample_rate;
				outputCodecCtx->channel_layout = inputCodecCtx->channel_layout;
				outputCodecCtx->channels = av_get_channel_layout_nb_channels(inputCodecCtx->channel_layout);
				outputCodecCtx->sample_fmt = codec->sample_fmts[0];
				outputCodecCtx->time_base = (AVRational){1 ,outputCodecCtx->sample_rate };
			}
			if (outputCtx->oformat->flags & AVFMT_GLOBALHEADER) {
				outputCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
			}
			ret = avcodec_open2(outputCodecCtx, codec, NULL);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Open Codec Encoder %s Open Stream#%u Failure\n", output, i);
				ret = AVERROR(ENOMEM);
				goto end;
			}
			ret = avcodec_parameters_from_context(out_stream->codecpar, outputCodecCtx);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Copy Codec Context %s Open Stream#%u Failure\n", output, i);
				ret = AVERROR(ENOMEM);
				goto end;
			}
			out_stream->time_base = outputCodecCtx->time_base;
			streamCtx[i].enc_ctx = outputCodecCtx;
		
		} else if (inputCodecCtx->codec_type == AVMEDIA_TYPE_UNKNOWN) {
			ret = AVERROR_INVALIDDATA;
			goto end;
		} else {
			/** Must Remuxed */
			ret = avcodec_parameters_copy(out_stream->codecpar ,in_stream->codecpar);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Copy Codec Context %s Open Stream#%u Remuxed Failure\n", output, i);
				ret = AVERROR(ENOMEM);
				goto end;
			}
			out_stream->time_base = in_stream->time_base;
		}
	
	}
	av_dump_format(outputCtx, 0, output, 1);
	if (!(outputCtx->oformat->flags & AVFMT_NOFILE)) {
		ret = avio_open(&outputCtx->pb, output, AVIO_FLAG_READ_WRITE);
		if (ret < 0) {
			av_log(NULL, AV_LOG_ERROR, "Ouput File %s Open Stream Failure\n", output);
			goto end;
		}
	}
	ret = avformat_write_header(outputCtx ,NULL);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Write Header\n");
		goto end;
	}

	//Init Filter
	const char* filterDescr = "overlay=100:100";

	AVFrame* srcFrame[2];
	AVFrame* inputFrame[2];
	AVFrame* filterFrame;
	AVFilterInOut* inputs;
	AVFilterInOut* outputs;
	AVFilterGraph* filterGraph = NULL;
	AVFilterContext* inputFilterContext[2];
	AVFilterContext* outputFilterContext = NULL;

	srcFrame[0] = av_frame_alloc();
	srcFrame[1] = av_frame_alloc();
	inputFrame[0] = av_frame_alloc();
	inputFrame[1] = av_frame_alloc();
	filterFrame = av_frame_alloc();

	filterGraph = avfilter_graph_alloc();
	if (!filterGraph) {
		av_log(NULL ,AV_LOG_ERROR ,"Filter Graph Alloc Fialure.\n");
		ret = AVERROR(ENOMEM);
		goto end;
	}
	avfilter_graph_parse2(filterGraph ,filterDescr ,&inputs ,&outputs);
	char args[512];
	memset(args ,0 ,sizeof(args));
	AVFilterContext* padFIlterContext = inputs->filter_ctx;
 	const AVFilter* filter = avfilter_get_by_name("buffer");
	AVCodecContext* codecContext = streamCtx[0].dec_ctx;
	sprintf_s(args ,sizeof(args),
		"video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
		codecContext->width ,codecContext->height ,codecContext->pix_fmt,
		codecContext->time_base.num ,codecContext->time_base.den / codecContext->ticks_per_frame ,
		codecContext->sample_aspect_ratio.num ,codecContext->sample_aspect_ratio.den);
	fprintf(stderr,
		"\nInput0 video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d \n",
		codecContext->width, codecContext->height, codecContext->pix_fmt,
		codecContext->time_base.num, codecContext->time_base.den / codecContext->ticks_per_frame,
		codecContext->sample_aspect_ratio.num, codecContext->sample_aspect_ratio.den);
	ret = avfilter_graph_create_filter(&inputFilterContext[0] ,filter ,"MainFrame" ,args ,NULL ,filterGraph);
	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Filter Config Main Frame Failure\n");
		goto end;
	}
	ret = avfilter_link(inputFilterContext[0] ,0 ,padFIlterContext ,inputs->pad_idx);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Filter Link Main Frame Failure\n");
		goto end;
	}

	char padArgs[512];
	memset(padArgs, 0, sizeof(padArgs));
	AVFilterContext* padNextFilterContext = inputs->next->filter_ctx;
	const AVFilter* nextFilter = avfilter_get_by_name("buffer");
	AVCodecContext* padCodecContext = inputCodecCtx[1];
	sprintf_s(padArgs, sizeof(padArgs),
		"video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
		padCodecContext->width, padCodecContext->height, padCodecContext->pix_fmt,
		padCodecContext->time_base.num, padCodecContext->time_base.den / padCodecContext->ticks_per_frame,
		padCodecContext->sample_aspect_ratio.num, padCodecContext->sample_aspect_ratio.den);
	fprintf(stderr,
		"\nInput1 video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d \n",
		padCodecContext->width, padCodecContext->height, padCodecContext->pix_fmt,
		padCodecContext->time_base.num, padCodecContext->time_base.den / padCodecContext->ticks_per_frame,
		padCodecContext->sample_aspect_ratio.num, padCodecContext->sample_aspect_ratio.den);
	ret = avfilter_graph_create_filter(&inputFilterContext[1], nextFilter, "OverlayFrame", padArgs, NULL, filterGraph);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Filter Config Overlay Frame Failure\n");
		goto end;
	}
	ret = avfilter_link(inputFilterContext[1], 0, padNextFilterContext, inputs->next->pad_idx);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Filter Link Overlay Frame Failure\n");
		goto end;
	}

	AVFilterContext* padOutputFilterContext = outputs->filter_ctx;
	const AVFilter* outputFilter = avfilter_get_by_name("buffersink");
	ret = avfilter_graph_create_filter(&outputFilterContext ,outputFilter ,"output" ,NULL ,NULL ,filterGraph);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Filter Config Output Failure\n");
		goto end;
	}
	ret = avfilter_link(padOutputFilterContext, outputs->pad_idx, outputFilterContext, 0);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Filter Link Output Failure\n");
		goto end;
	}
	avfilter_inout_free(&inputs->next);
	avfilter_inout_free(&inputs);
	avfilter_inout_free(&outputs);
	ret = avfilter_graph_config(filterGraph ,NULL);
	if (ret < 0) {
		av_log(NULL, AV_LOG_ERROR, "Filter Graph Config Failure\n");
		goto end;
	}

	// Deocde
	int gotFrame = 0;
	int16_t streamIndex;
	int16_t gotOutput;
	AVPacket packet;

	ret = 1;
	
	while (ret) {
		packet.size = 0;
		packet.data = NULL;
		av_init_packet(&packet);
		ret = av_read_frame(inputCtx[1] ,&packet);
		int gotFrame = 0;
		ret = avcodec_decode_video2(inputCodecCtx[1] ,srcFrame[1] ,&gotFrame ,&packet);
		if (ret >= 0 && gotFrame != 0) {
			srcFrame[1]->pts = packet.pts;
			break;
		}
	}

	while (1) {
		packet.size = 0;
		packet.data = NULL;
		av_init_packet(&packet);
		ret = av_read_frame(inputCtx[0], &packet);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_INFO ,"Read Input Frame End.\n");
			break;
		} else {
			av_log(NULL, AV_LOG_DEBUG, "Read Frame Success.\n");
		}
		streamIndex = packet.stream_index;
		ret = avcodec_decode_video2(streamCtx[streamIndex].dec_ctx , srcFrame[0] ,&gotFrame ,&packet);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Decoding Failed.\n");
		}
		if (gotFrame) {
			av_log(NULL ,AV_LOG_DEBUG ,"Push Decode Frame To Filters\n");
			srcFrame[0]->pts = packet.pts;
			av_frame_ref(inputFrame[0] ,srcFrame[0]);
			ret = av_buffersrc_add_frame_flags(inputFilterContext[0] , srcFrame[0] ,AV_BUFFERSRC_FLAG_PUSH);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Frame#0 Add Frame Failed.\n");
				break;
			}
			srcFrame[0]->pts = srcFrame[0]->best_effort_timestamp;
			srcFrame[1]->pts = srcFrame[0]->pts;
			//av_frame_ref(inputFrame[1], srcFrame[1]);
			ret = av_buffersrc_add_frame_flags(inputFilterContext[1], srcFrame[1], AV_BUFFERSRC_FLAG_PUSH);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Frame#1 Add Frame Failed.\n");
				break;
			}
			ret = av_buffersink_get_frame_flags(outputFilterContext,filterFrame ,AV_BUFFERSINK_FLAG_NO_REQUEST);
			if (ret < 0) {
				av_log(NULL, AV_LOG_ERROR, "Frame#0 Output Frame Failed.\n");
				av_frame_unref(filterFrame);
				break;
			}
			else {
				av_log(NULL, AV_LOG_ERROR, "Frame#0 Output Frame Success.\n");
			}
			packet.size = 0;
			packet.data = NULL;
			av_init_packet(&packet);
			ret = avcodec_encode_video2(streamCtx[streamIndex].enc_ctx,&packet ,filterFrame ,gotOutput);
			if (ret >=0 && gotOutput) {
				ret = av_write_frame(outputCtx ,&packet);
			}
			av_frame_unref(filterFrame);
			
		}

		av_packet_unref(&packet);
	}
	av_write_trailer(outputCtx);
	if (inputCtx[0] != NULL){
		avformat_close_input(&inputCtx[0]);
	}
	if (outputCtx != NULL) {
		for (int i = 0; i < outputCtx->nb_streams; i++) {
			avcodec_free_context(&streamCtx[i].enc_ctx);
			avcodec_free_context(&streamCtx[i].dec_ctx);
		}
	}
	avformat_close_input(&outputCtx);

	fprintf(stderr ,"\nFRAME OVERLAY SUCCESS\n");
	return 0;
end:
	fprintf(stderr ,"\nFRAME OVERLAY FAILURE\n");
	return -1;
}
