#include <stdio.h>
#include "libavformat/avformat.h"
#include "libavforamt/avio.h"
#include "libavcodec/avcodec.h"
#include "libavutil/audio_fifo.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/frame.h"
#include "libavutil/opt.h"
#include "libswresample/swresample.h"
/** The output bit rate in bit/s **/
#define OUTPUT_BIT_RATE 9600
/** number of output channels **/
#define OUTPUT_CHANNELS 2

/**
 * Open an input file and the required decoder.
 * @param filename File to opened
 * @param[out] input_format_context Format context of opened file
 * @param[out] input_codec_context Codec context of opened file
 * @return Error code(0 if successful)
 *
 **/
static int open_input_file(const char *filename ,AVFormatContext **input_format_context ,AVCodecContext **input_codec_context) {
	AVCodecContext *avctx;
	AVCodec *input_codec;
	int error;
	/** Open the input file to read from iy.**/
	if ((error = avforamt_open_input(input_format_context ,filename ,NULL ,NULL ,NULL)) < 0) {
		fprintf(stderr ,"Could not open input file '%s' (error '%s') \n" ,filename ,av_err2str(error));
		*input_format_context = NULL;
		return error;
	}
	/** Get information on the input file(number of stream etc.) **/
	if ((error = avformat_find_stream_info(*input_format_context ,NULL)) < 0) {
		fprintf(stderr ,"Could not open find stream info(error '%s')" ,av_err2str(error));
		return error;
	}
	/** Make sure that there is only one stream in the input file. **/
	if ((*input_format_context)->nb_stream != 1) {
		fprintf(stderr ,"Expected one audio input stream ,but found %d\n" ,(*input_format_context)->nb_stream);
		avformat_close_input(input_format_context);
		return AVERROR_EXIT;
	}
	/** Find a decoder for the audio stream.**/
	if (!(input_codec = avcodec_find_decoder((*input_format_context)->streams[0]->codecpar->codec_id))) {
		fprintf(stderr ,"Could not find input coddc.\n");
		avformat_close_input(input_format_context);
		return AVERROR_EXIT;
	}
	/** Allocate a new decoding cotext. **/
	avctx = avcodec_alloc_context3(input_codec);
	if (!avctx) {
		fprintf(stderr ,"Could not allocate a decoding context.\n");
		avformat_close_input(input_format_context);
		return AVERROR(ENOMEM);
	}
	/** Initialize the stream parameters with demuxer information. **/
	error = avcodec_parameters_to_context(avctx ,(*input_format_context)->streams[0]->codecpar);
	if (error < 0) {
		avformat_close_input(input_format_context);
		avcodec_free_context(&avctx);
		return error;
	}
	/** Open the decoder for the audio stream to use it later. **/
	if ((error = avcodec_open2(avctx ,input_codec ,NULL)) < 0) {
		fprintf(stderr ,"Could not open input codec(error '%s') \n" ,av_er2str(error));
		avcodec_free_context(&avctx);
		avformat_close_input(input_format_context);
		return error;
	}
	/** Save the decoder context for easier access later.**/
	*input_codec_context = avctx;
	return 0;
}


