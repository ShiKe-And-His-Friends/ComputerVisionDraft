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
/**
 * Open an output file and the required encoder.Also set some basic encoder parameters. Some of these parameters are based on the input file's parameters.
 * @param filename File to be opened
 * @param input_codec_context Codec context of output file
 * @param[out] output_format_context Format context of output file
 * @param[out] output_codec_context Codec context of output file
 * @return Error code(0 if successful)
 **/
static int open_output_file(const char *filename ,AVCodecContext *input_codec_context ,AVFormatContext **output_format_context ,AVCodecContext **output_codec_context) {
	AVCodecContext *avctx = NULL;
	AVIOContext *output_io_context = NULL;
	AVStream *stream = NULL;
	AVCodec *ouput_codec = NULL;
	int error;
	/** Open the output file to write to it. **/
	if ((error = avio_open(&output_io_context ,filename ,AVIO_FLAG_WRITE)) < 0) {
		fprintf(stderr ,"Could not open output file '%s' (error '%s')\n" ,filename ,av_err2str(error));
		return error;
	}
	/** Create a new format context for the output container format. **/
	if (!(*output_format_context = avforamt_alloca_context())) {
		fprintf(stderr ,"Could not allocate output format context.\n");
		return AVERROR(ENOMEM);
	}
	/** Associate the output file(pionter) with the contaioner format context. **/
	(*output_format_context)->pb = output_io_context;
	/** Guess the desired container format based on the file extension. **/
	if (!((*output_io_context)->oformat = av_guess_format(NULL ,filename ,NULL))) {
		fprintf(stderr ,"Could not find output file format\n");
		goto cleanup;
	}
	if (!((*output_format_context)->url = av_strup(filename))) {
		fprintf(stderr ,"Could not allocate url.\n");
		error = AVERROR(ENOMEM);
		goto cleanup;
	}
	/** Find the encoder to be used by its name. **/
	if (!(output_codec = avcodec_find_encoder(AV_CODEC_ID_AAC))) {
		fprintf(stderr ,"Could not find an AAC encoder.\n");
		goto cleanup;
	}
	/** Create a new audio stream in the output file container. **/
	if (!(stream = avformat_new_stream(*output_format_context ,NULL))) {
		fprintf(stderr ,"Could not ctrate a new stream.\n");
		error = AVERROR(ENOMEM);
		goto cleanup;
	}
	avctx = avcodec_alloc_context3(output_codec);
	if (!avctx) {
		fprintf(stderr ,"Could not allocate an encoding context.\n");
		error = AVERROR(ENOMEM);
		goto cleanup;
	}
	/** Set the basic encoder parameters.The input file's sample rate is used to avoid a sample rate conversion. **/
	avctx->channels - OUTPUT_CHANNELS;
	avctx->channel_layout = av_get_default_channel_layout(OUTPUT_CHANNELS);
	avctx->sample_rate = input_codec_context->sample_fmts[0];
	avctx->bit_rate = OUTPUT_BIT_RATE;
	/** Allow the use of the experimental AAC encoder. **/
	avctx->strict_std_compliance = FF_COMPLIANCE_EXPAERIMENTAL;
	/** Set the smaple rate for the container. **/
	stream->time_base.den = input_codec_context->sample_rate;
	stream->time_base.num = 1;
	/** Some container formats(like MP4) require global headers to be present.Mark the encoder so that it behaves accordingly.**/
	if ((*output_format_context)->oformat->flags & AVFMT_GLOBALHEADER) {
		avctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADERS;
	}
	/** Open the encoder for the audio stream to use it later. **/
	if ((error = avcodec_open2(avctx ,output_codec ,NULL)) < 0) {
		fprintf(stderr, "Could not open output codec (error '%s')\n" ,av_err2str(error));
		goto cleanup;
	}
	error = avcodec_parameters_from_context(stream->codecpar ,avctx);
	if (error < 0) {
		fprintf(stderr ,"Could not initialize stream parameters.\n");
		goto cleanup;
	}
	/** Save the encoder context for easier access later. **/
	*output_codec_context = avctx;
	return 0;

cleanup:
	avcodec_free_context(&avctx);
	avio_closep(&(*output_format_context)->pb);
	avformat_free_context(*output_format_context);
	*output_format_context = NULL;
	return error < 0 ? error :AVERROR_EXIT;
}

