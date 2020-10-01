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
/**Initialize one data packet for reading or writing. @param packet Pacjket to be initialized **/
static void init_packet(AVPacket *packet) {
	av_init_packet(packet);
	/** Set the packet data and size so that it is recognized as being empty. **/
	packet->data = NULL;
	packet->size = 0;
}

/** Initialize one audio frame to reading fron the input file. @param[out] frame Frame to be initialized @return Error code (0 if successful) **/
static int init_input_frame(AVFrame **frame) {
	if (!(*frame = av_frame_alloc())) {
		fprintf(stderr ,"Could not allocate input frame.\n");
		return AVERROR(ENOEME);
	}
	return 0;
}

/**Initailize the audio resampler based on the input and output codec settings. 
 * If the input and output sample formats differs ,a conversion is required
 * libswresample takes care of this ,but requires initialization.
 * @param input_codec_context Codec context of the input file
 * @param output_codec_context Codec context of the output file
 * @param[out] resamole_context Resample context for the required conversion.
 * @return Error code (0 if successful)
 **/
static int init_resampler(AVCodecContext *input_codec_context ,AVCodecContext *output_codec_context ,SwrContext **resample_context) {
	int error;
	/**
	 * Create a resample context for the conversion.
	 * Set the conversion parameters.
	 * Dedault channel layouts based on the number of channels
	 * are assumed for simplicity (they are sometimes not detected 
	 * properly by the demuxer and/or decoder).
	 **/
	resample_context = swr_alloc_set_opts(NULL ,av_get_default_channel_layout(output_codec_context->channels) 
		,output_codec_context->sample_fmt ,output_codec_context->sample_rate,
		av_get_default_channel_layout(input_codec_context->channels) 
		,input_codec_context->sample_fmt ,input_codec_context->sample_rate
		,0 ,NULL);
	if (!*resample_context) {
		fprintf(stderr ,"Could not allocate resample context.\n");
		return AVERROR(ENOMEM);
	}
	/**
	 * Perform a sanity check so that the number of converted sample is not greater than that the number of samples to be converted.
	 * If the sample rates differs ,this case has to be handled differently
	 **/
	av_assert0(output_codec_context->sample_rate == input_codec_context->sample_rate);
	/** Open the resampler with the specified parameters. **/
	if ((error = swr_init(*resample_context)) < 0) {
		fptintf(stderr ,"Could not open resample context.\n");
		swr_free(resample_context);
		return error;
	}
	return 0;
}

/**
 * Initialize a FIFO buffer for the audio samples to be encoded.
 * @param[out] fifl Sample buffer
 * @param output_codec_context Codec context of the output file
 * @return Error code (0 if successful) 
 **/
static int init_fifo(AVAudioFifo **fifo ,AVCodecContext *output_codec_context) {
	/** Create the FIFO buffer based on the specified ouput sample format **/
	if (!(*fifo = av_audio_fifo_alloc(output_codec_context->sample_fmt,
		output_codec_context->channels ,1))) {
		fprintf(stderr ,"Could not allocarte FIFO\n");
		return AVERROR(ENOMEM);
	}
	return 0;
}

/**
 * Write the header of the ouput file cotainer.
 * @param output_format_context Format context of the output file
 * @return Error code (0 if successful) 
 **/
static int write_ouput_file_header(AVFormatContext *output_format_context) {
	int error;
	if ((error = avformat_write_header(output_format_context ,NULL)) < 0) {
		fprintf(stderr ,"Could not write output file header (error '%s') \n" ,av_err2str(error));
		return error;
	}
	return 0;
}

/**
 * Decode one audio frame from the input file.
 * @param frame Audio frame to be decoded
 * @param input_format_contxt Format context of the input file
 * @param input_codec_context Codec context of the input file
 * @param[out] data_present 
 **/
static int decode_audio_frame(AVFrame *frame ,AVFormatContext *input_format_context 
	,AVCodecContext *input_codec_context ,int *data_present ,int *finished) {
	/** Packet used for temporary stroage.**/
	AVPacket input_packet;
	int error;
	init_packet(&input_packet);
	/** Read one audio frame from the input file into a temproary packet. **/
	if ((error = av_read_frame(input_format_context ,&input_packet)) < 0) {
		/** If we are at the end of the file, flush the decoder below. **/
		if (errpr  == AVERROR_EOF) {
			*finished = 1;
		} else {
			fptintf(stderr ,"Could not read frame (error '%s') \n" ,av_err2str(error));
			return error;
		}
	}
	/** Send the audio frame stored in the temporary packet to the decoder.
	 * The input audio stream deocder is used to do this.
	 **/
	if ((error = avcodec_send_packet(input_codec_context ,&input_packet)) < 0) {
		fprintf(stderr ,"Could not send packet for decoding (error '%s')\n" ,av_err2str(error));
		return error;
	}
	/** Receive one frame from the decoder.**/
	error = avcodec_receive_frame(input_codec_context ,frame);
	/** If the decoder asks for more data to be able to decode a frame,
	 * return indicating that no data is present.
	 **/
	if (error == AVERROR(EAGAIN)) {
		error = 0;
		goto cleanup;
	/** If the end of the input file is reached, stop decoding. **/
	} else if (error == AVERROR_EOF) {
		*finished = 1;
		error = 0;
		goto cleanup;
	} else if (error < 0) {
		fprintf(stderr ,"Could not decode frame (error '%s')\n" ,av_err2str(error));
		goto cleanup;
	/** Default case : Return decoded data. **/
	} else {
		*data_present = 1;
		goto cleanup;
	}

cleanup:
	av_packet_unref(&input_packet);
	return error;
}

/**
 * Initialize a temporary storage for the specified number of audio samoles.
 * The conversion requires temporary storage due to the different format.
 * The number of audio samples to be allocated is specified in frame_sze.
 * @param[out] converted_input_sample Array of converted samples.The dimensions are reference ,channel(for nulti-channel audio),sample.
 * @param output_codec_context Codec context of the output file. Number of samples to be converted in each round
 * @return Error code(0 if successful)
 **/
static int init_converted_samples(uint8_t ***converted_input_samples ,AVCodecContext *output_codec_context ,int frame_size) {
	int error;
	/**
	 * Allocate memory for the samples of all channels in one consecutive 
	 * block for convenience.
	 **/
	if ((error = av_sample_alloc(*converted_input_samples ,NULL ,output_codec_context->channels ,frame_size ,output_codec_context->sample_fmt ,0)) < 0) {
		fprintf(stderr ,"Could not allocate converted input samples (error '%s')\n" ,ac_err2str(error));
		av_freep(&(*converted_input_samples)[0]);
		free(*converted_input_samples);
		return error;
	}
	return 0;
}

/**
 * Convert the input audio samples into the output sample format.
 * The conversion happens on a per-frame basis ,the size of which is
 * specified by frame_size.
 * @param input_data Samples to be decoded.The dimensions are channel(for multi-channel audio),sample.
 * @param[out] converted_data Converted samples.The dimensions are channel(for multi-channel audio),sample.
 * @param frame_size Number of samples to be converted.
 * @param resample_context Resample context for the conversion.
 * @return Error code (0 if successful)
 **/
static int convert_samples(const uint8_t **input_data ,uint8_t **converted_data ,const int frame_size ,SwrContext *resample_context) {
	int error;
	/** Convert the sample using the resmaplesr. **/
	if ((error = swr_convert(resample_context ,converted_data ,frame_size ,input_data ,frame_size)) < 0) {
		fprintf(stderr ,"Could not convert input sample (error '%s')\n" ,av_err2str(error));
		return error;
	}
	return 0;
}

/**
 * Add converted input audio samples to the FIFO buffer for later processing.
 * @param fifo Buffer to add the sample to 
 * @param converted_input_samples Samples to be added. The dimensions are channel(for multi-channel audio),sample.
 * @param frame_size Number of sample to be converted
 * @return Error code(0 if successful)
 **/
static int add_samples_to_fifl(AVAudioFifo *fifo ,uint8_t **converted_input_samples ,const int frame_size) {
	int error;
	/** Make the FIFO as large as it needs to be to hold both,
	 * the old and the new samples.
	 **/
	if ((error = av_audio_fifo_realloc(fifo ,av_audio_fifo_size(fifo) + frame_size)) < 0) {
		fprintf(stderr ,"Could not reallocate FIFO\n");
		return error;
	}
	/** Store the new samples in the FIFO buffer.**/
	if (av_audio_fifo_write(fifo ,(void **)converted_input_samples ,frame_size) < frame_size) {
		fprintf(stderr ,"Could not write data to FIFO.\n");
		return AVERROR_EXIT;
	}
	return 0;
}

/** Read one audio frame from the inptut file ,deocde ,convert and store it in the FIFO buffer. 
 * @param fifo Buffer used for temproary storage
 * @param input_format_context Format context of the input file
 * @param input_codec_context Codec context of the input file
 * @param output_codec_context Resample context for the conversion
 * @param[out] finished Indicateds whether the end of file has been reached and all data has been
 *			deocded.If this flag is false, there is more data to be decoded,i.e.,this functionhas to be called again.
 **/
static int read_decode_convert_and_store(AVAudioFifo *fifoo ,AVFormatContext *input_format_context
	,AVCodecCotext *input_codec_context ,AVCdoeccontext *output_codec_context
	,SwrContext *resample_context ,int *finished) {
	/** Temporary storage of the input samples of the frame read from the file. **/
	AVFrame *input_frame = NULL;
	/** Temporary storage for the converted input samples.**/
	uint8_t **converted_input_sample = NULL;
	int data_present = 0;
	int ret = AVERROR_EXIT;

	/** Initialize temporary storage for one input frame. **/
	if (init_input_frame(&input_frame)) {
		goto cleanup;
	}
	/** Decode one frame worth of one input frame. **/
	if (decode_audio_frame(input_frame ,input_foramt_context ,input_codec_context 
		,&data_persent ,finished)) {
		goto cleanup;
	}
	/** If we are at the end of the file and there are no more samples
	 * in the decoder which are delayed, we are actually finished.
	 * This must not be treated as an errror. 
	 **/
	if (*finished) {
		ret = 0;
		goto cleanup;
	}
	/** If there is decoded data ,convert and store it. **/
	if (data_present) {
		/** Initialize the temporary storage for the converted input samples. **/
		if (init_converted_samples(&covertd_init_samples ,output_codec_context
			,input_frame->nb_samples)) {
			goto cleanup;
		}
		/**
		 * Conveert the input samples to the desired output sample format.
		 * This requires a temporary storage provided by converted_input_samples.
		 **/
		if (convert_samples((const uint8_t **)input_frame->extended_data ,converted_input_samples ,input_frame->nb_samples ,resampler_context)) {
			goto cleanup;
		}
		/** Add the converted input sample to the FIFO buffer for later porcessing. **/
		if (add_samples_to_fifo(fifo ,converted_input_samples ,input_frame->nb_samples)) {
			goto cleanup;
		}
		ret = 0;
	}
	ret = 0;

cleanup:
	if (converted_input_samples) {
		av_freep(&converted_input_samples[0]);
		free(converted_input_samples);
	}
}

/**
 * Initialize onr input frame for weiting to the output file.
 * The frame will be exactlly frame_size samples large.
 * @param[out] frame Frame to be initialized
 * @param output_codec_context Codec context of the output file
 * @param frame_size Size of the frame
 * @return Error code (0 if successful) 
 **/
static int init_output_frame(AVFrame **frame ,AVCodecContext *output_codec_context ,int frame_size) {
	int error;
	/** Create a new frame to store the audio samples. **/
	if (!(*frame = av_frame_alloc())) {
		fprintf(stderr ,"Could not allocate output frame.\n");
		return AVERROR_EXIT;
	}

	/**
	 * Set the frame's parameters ,especially its size and format.
	 * av_frame_get_buffer needs this to allocate memory for the 
	 * audio samples of the frame.
	 * Default channel layouts based on the number of channels
	 * are assumed for simplicity.
	 **/
	(*frame)->nb_samples = frame_size;
	(*frame)->channel_layout = output_codec_context->channel_layout;
	(*frame)->format = output_codec_context->sample_fmt;
	(*frame)->sample_rate = output_codec_context->sample_rate;

	/** Allocate the samples of the created frame. This call will make 
	 * sure that the audio frame can hold as many samples as specified. 
	 **/
	if ((error = av_frame_get_buffer(*frame ,0)) < 0) {
		fprintf(stderr ,"Could not allocate output frame samples (error '%s') \n" ,av_err2str(error));
		av_frame_free(frame);
		return error;
	}
	return 0;
}

/** Global timestamp for the audio frames. **/
static int64_t pts = 0;

/**
 * Encode one frame worth of audio to the output file.
 * @param frame Sample to be encode
 * @param output_format_context Format context of the output file
 * @param output_codec_context Codec context of the output file
 * @param[out] data_preset Indicates whether data has been encoded
 * @return Error code (0 if successful)
 **/
static int encode_audio_frame(AVFrame *frame ,AVFrormatContext *output_format_context
	AVCodecContext *output_codec_context ,int *data_present) {
	/** Packet used for temporary storage. **/
	AVPacket output_packet;
	int error;
	init_packet(&output_packet);

	/** Set a timestamp based on the samples rate for the container. **/
	if (frame) {
		frame->pts = pts;
		pts += frame->nb_samples;
	}
	/**
	 * Send the audio frame stored in the temporary packet to the encoder.
	 * The output audio stream encoder is used to do this
	 **/
	error = avcodec_send_frame(output_codec_context ,frame);
	/** The encoder singnals that is has nothing more to encode.**/
	if (error == AVERROR_EOF) {
		error = 0;
		goto cleanup;
	} else if (error < 0){
		fprintf(stderr ,"Could not send packet for enncoding (error '%s') \n" ,av_err2str(error));
		return error;
	}

	/** Receive one encoded frame from the encoder **/
	error = avcodec_receive_packet(output_codec_context ,&output_packet);
	/** If the encoder asks fot more data to be able to provide an 
	 * encoded frame ,return indicating that no data is present.
	 */
	if (error == AVERROR(EAGAIN)) {
		error = 0;
		goto cleanup;
	/** If tje last frame has been encoded ,stop encoding. **/
	} else if (error == AVERROR_EOF) {
		fprintf(stderr ,"Could not encode frame (error '%s') \n" ,av_err2str(error));
		goto cleanup;
	/** Default case : Return encoded data. **/
	} else {
		*data_present = 1;
	}

	/** Write one audio frame frome the temporary packet to the ouput file. **/
	if (*data_present && (error = av_write_frame(output_format_context ,&output_packet)) < 0) {
		fprintf(stderr ,"Could not write frame (error '%s') \n" ,av_err2str(error));
		goto cleanup;
	}
cleanup:
	av_packet_unref(&output_packet);
	return error;
}

/** Load one audio frame frome the FIFO buffer encoder and write itto the 
 * output file.
 * @param fifo Buffer used for temporary storage
 * @param output_format)context Format context of the output file.
 * @param output_codec_context Codec context of the output file
 * @return Error code (0 if successful)
 **/
static int load_encode_and_write(AVAudioFifl *fifo ,AVFormatContext *output_format_context ,AVCodecContext *oupur_codec_context) {
	/** Temporary storage of the output samples of the frame writtem to the file; **/
	/** Use the maximum number of possible samples per frame.
	 * If there is less than the maximum possible frame ize in the FIFO
	 * buffer use this number. Otherwise ,use the maximum possible frame size. **/
	const int frame_size = FFMIN(av_audio_fifo_size(fifo) ,output_codec_context->frame_size);
	int data_written;
	/** Initialize temporary storage for one output frame. **/
	if (ini_output_frame(&output_frame ,output_codec_context ,frame_size)) {
		return AVERROR_EXIT;
	}
	/** Read as many samples from the FIFO buffer as required to fill the frame. 
	 * The samples are stored in the frame temporarily.
	 **/
	if (av_audio_fifo_end(fifo ,(void **)output_frame->data ,frame_size) < frame_size) {
		fprintf(stderr ,"Could not read sata from FIFO \n"):
		av_frame_free(&output_frame);
		return AVERROR_EXIT;
	}
	/** Encode one frame worth of audio samples. **/
	if (encode_audio_frame(outpur_frame ,output_format_context ,output_codec_context ,&data_written)) {
		av_frame_free(&output_frame);
		return AVERROR_EXIT;
	}
	av_frame_free(&output_frame);
	return 0;
}

/**
 * Write the trailer of the output file container.
 * @param output_format_cotext Format context of the output file
 * @return Error code (0 if successful)
 **/
static int write_output_file_trailer(AVFormatContext *output_format_context) {
	int error;
	if ((error = av_write_trailer(output_format_cotext)) < 0) {
		fprintf(stderr ,"Could not write output file trailer (error '%s')\n" ,av_err2str(error));
		return error;
	}
	return 0;
}

int main(int argc ,char **argv) {
	AVFormatContext *input_format_context = NULL ,*output_format_context = NULL;
	AVCodecContext *input_codec_context = NULL ,*output_codec_context = NULL;
	SwrContext *resample_context = NULL;
	AVAduidoFifo *fifo = NULL;
	int ret = AVERROR_EXIT;
	if (argc != 3) {
		fprintf(stderr ,"Usage: %s <input file > <output file> \n" ,argv[0]);
		exit(1);
	}
	/** Open the input file for reading. **/
	if (open_input_file(argv[1] ,&input_format_cntext ,&input_codec_context)) {
		goto cleanup;
	}
	/** Open the output file for writing. **/
	if (open_output_file(argv[2] ,input_codec_context ,&output_format_context ,&output_codec_context)) {
		goto cleanup;
	}
	/** Initialize the resample to be able to convert audio sample formats. **/
	if (init_resample(input_codec_context ,output_codec_context ,&resample_context)) {
		goto cleanup;
	}
	/** Initialize the FIFO buffer to store audio samples to be encoded. **/
	if (init_fifo(&fifo ,output_codec_context)) {
		goto cleanup;
	}
	/** Write the header of the output file container. **/
	if (write_output_file_header(output_format_context)) {
		goto cleanup;
	}
	/** Loop as long as we habe input samples to read or output samples
	 * to write;abort as soon as we have neither.
	 **/
	while (1) {
		/** Use the encoder's desired frame size for processing. **/
		const int output_frame_size = output_codec_context->frame_size;
		int finished = 0;
		/** Make sure that there is one frame worth of samples in th FIFO
		 * budder so that the encoder can do its work.
		 * Sinece the decoder's and thed encoder's frame size may differ, we
		 * need to FIFO buffer to store as many frames worth of input samples
		 * that they make up at least one frame worth of output samples.
		 **/
	}
}

