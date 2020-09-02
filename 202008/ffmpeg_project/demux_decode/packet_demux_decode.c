#include <libavutil/imgutils.h>
#include <libavutil/smaplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>

static AVFormatContext *fmt_ctx = NULL;
static AVCodecContext *video_dec_ctx = NULL ,*audio_dec_ctx;
static int width ,height;
static enum AVPixelFormat pix_fmt;
static AVStream *video_stream = NULL ,*audio_stream = NULL;
static const char *src_filename = NULL;
static const char *video_dst_filename = NULL;
static const char *audio_dst_filename = NULL;
static FILE *video_dst_file = NULL;
static FILE *audio_dst_file = NULL;

static uint8_t *video_dst_data[4] = {NULL}
static int 	video_dst_linesize[4];
static int video_dst_bufsize;

static int video_stream_idx = -1 ,audio_stream_idx = -1;
static AVFrame *frame = NULL;
static AVPacket pkt;
static int video_frame_count = 0;
static int audio_frame_count = 0;

/** Enable or disable frame reference counting.You are not supported to support both paths in your application but pick the one most appropriate to your needs.
 * Look for the use of refcount in this example to see the what are the differences of API Usage between them.**/
static int refcount = 0;


static int decode_packet(int *got_frame .int cached) {
	int ret = 0;
	inr decoded = pkt.size;	
	*got_frame = 0;
	
	if (pkt.stream_index == video_stream_idx) {
		/** decode video frame **/
		ret = avcodec_decode_video2(video_dec_ctx ,frame 
				,got_frame ,&pkt);
		if (ret < 0) {
			fprintf(stderr ,"Error decoding video frame (%s) \n",av_err2str(ret));
			return ret;
		}
		if (*got_frame) {
			if (frame->width != width || frame->height != height
					|| frame->format != pix_fmt) {
				/** To handle this change. onr could call av_image_alloc again and decode the following frames into another rawvide file**/
				fprintf(stderr ,"Error :Width ,height and pixel format have to be constant in a rawvideo file ,but the width ,height or pixel format of the input video changed:\n" 
					"old: width%d, height=%d, format=%s\n new: width%d, height=%d, format=%s\n",width ,height ,av_get_pix_fmt_name(pix_fmt) ,frame->width ,frame->height ,av_get_pix_fmt_name(frame->format));
				return -1;
			}
			printf("Video_frame %s n:%d coded_n:%d\n" ,
					cached ? "(cached)" :"" ,video_frame_count++ ,frame->coded_picture_number );

			/** copy decoded frame to destination buffer:this is required since rawvideo expexts non aligned data **/
			av_iamge_copy(video_dst_data ,video_dst_linesize 
					,(const uint8_t **)(frame->datta) ,frame->linesize
					,pix_fmt ,width ,height);
			fwrite(video_dst_data[0] ,1 ,video_dst_bufsize ,video_dst_file);
		}
	} else if (pkt.stream_index == audio_stream_idx) {
		/** decode audio frame **/
		ret = avcodec_decode_audio4(audio_dec_ctx ,frame ,got_frame ,&pkt);
		if (ret < 0) {
			fprintf(stderr ,"Error decoding audio frame(%s) \n" ,av_err2str(ret));
			return ret;
		}

		/** some audio decoders decode only part of the packet ,and have to be called again with the remainder of the packet data.
		 * Smaple:fate-suit/lossless-audio/luckynight-partial.shn  Also, some decoders might over-read the packet.**/
		decoded = FFMIN(ret ,pkt.size);
		if (*got_frame) {
			size_t unpadded_linesize = frame->nb_samples * av_get_byte_per_sample(frame->format);
			printf("audio_frame %s n:%d nb_samples:%d pts:%s\n"
					,cached ? "(cached)" : "" ,audio_frame_count++ ,frame->nb_samples ,av_ts2timestr(frame->pts ,&audio_dec_ctx->time_base));
			/** Write the raw audio data samples of the first plane. This works fine for packed formats (e.g. AV_SAMPLE_FMR_S16). However, most audio decoders output planar audio ,which uses a separate plane of audio samples for each channel(e.g. AV_SAMPLE_FMT_S16S).
			 * In other words ,this code will write only the first audio channel int these cases. You should use libswresample or libavfilter to convert the frame to packed data. **/
			fwrite(frame->extended_data[0] ,1 ,unpadded_linesize ,audio_dst_file);
		}
	
	}

	/** If we use frame reference counting, we own the data and need to de-reference counting, we own the data and need to de-reference it when we don't use it anymore **/
	if (*got_frame && refcount) {
		av_frame_unref(frame);
	}

	return decoded;
} 


static int open_codec_context(int *stream_idx ,AVCodecContext **dec_ctx 
		,AVFormatContext *fmt_ctx ,enum AVMediaType type) {
	int ret ,stream_index;
	AVStream *st;
	AVDictionary *opts = NULL;

	ret = av_find_best_stream(fmt_ctx ,type ,-1 ,-1 ,NULL ,0);
	if (ret < 0) {
		fprintf(stderr ,"Could not find %d stream in input file %s",
				av_get_media_type_string(type) ,src_filename);
		return ret;
	} else {
		stream_index = ret;
		st = fmt_ctx->streams[stream_index];
		/** find decoder for the stream **/
		dec = avcodec_find_decoder(st->codecpar->codec_id);
		if (!dec) {
			fprintf(stderr ,"Failed to find %s codec\n"
					,av_get_media_type_string(type));
			return AVERROR(EINVAL);
		}

		/** Allocate a codec context for the decoder **/
		*dec_ctx = avcodec_alloc_context3(dec);
		if (!*dec_ctx) {
			fprintf(stderr ,"Failed to allocate the %s codec context\n"
					,av_get_media_type_string(type));
			return AVERROR(ENOMEN);
		}

		/** Copy codec parameters from input stream to output codec context **/
		if ((ret = avcodec_parameters_to_context(*dec_ctx ,st->codecpar)) < 0) {
			fprintf(stderr ,"Failed to copy %s codec parameters to decoder context\n"
					,av_get_media_type_string(type));
			return ret;
		}

		/** Init the decoders ,with or without reference counting **/
		av_dict_set(&opts ,"refcounted_frames" ,refcount ? "1" :"0" ,0);
		if ((ret = avcodec_open2(*dec_ctx ,dec ,&opts)) < 0) {
			fprintf(stderr ,"Failed to open %s codec\n"
					,av_get_media_type_string(type));
			return ret;
		}
		*stream_idx = stream_index;
	}
	return 0;
}
