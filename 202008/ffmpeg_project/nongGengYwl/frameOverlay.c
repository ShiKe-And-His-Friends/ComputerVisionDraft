#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <stdio.h>

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
	ret = avformat_find_stream_info(inputCtx[0] ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Info Input File %s Failure\n" ,input[0]);
		goto end;
	}
	av_dump_format(inputCtx[0] ,0 ,input[0] ,0);
	enum AVCodecID codecId = inputCtx[0]->streams[0]->codecpar->codec_id;
	AVCodec* codec = avcodec_find_decoder(codecId);	
	if (!codec) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Video# %d failure\n" ,codecId);
		goto end;
	}
	inputCodecCtx[0] = avcodec_alloc_context3(codec);
	if (!inputCodecCtx[0]) {	
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Context Video# %d failure\n" ,codecId);
		goto end;
	}
	ret = avcodec_open2(inputCodecCtx[0] ,codec ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Output Codec For File %s Failure\n" ,input[0]);
		goto end;
	}
	ret = avcodec_parameters_to_context(codec ,);

	inputCtx[1] = avformat_alloc_context();
	inputCtx[1]->interrupt_callback.callback = NULL;
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
	enum AVCodecID codecIdPic = inputCtx[1]->streams[0]->codecpar->codec_id;
 	AVCodec* codecPic = avcodec_find_decoder(codecIdPic);	
	if (!codecPic) {
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Video# %d failure\n" ,codecIdPic);
		goto end;
	}
	inputCodecCtx[1] = avcodec_alloc_context3(codecPic);
	if (!inputCodecCtx[1]) {	
		av_log(NULL ,AV_LOG_ERROR ,"Find Decoder Context Video# %d failure\n" ,codecId);
		goto end;
	}
	ret = avcodec_open2(inputCodecCtx[1] ,codecPic ,NULL);

	if (ret < 0) {
		av_log(NULL ,AV_LOG_ERROR ,"Open Output Codec For File %s Failure\n" ,input[1]);
		goto end;
	}
	//TODO stream codec info

	// Open Output File
	ret = avformat_alloc_output_context2(&outputCtx ,NULL ,"mpegts" ,output);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Ouput File %s Open Failure\n" ,output);
		goto end;
	}
	ret = avio_open2(&outputCtx->pb ,output ,AVIO_FLAG_READ_WRITE ,NULL ,NULL);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Ouput File %s Open Stream Failure\n" ,output);
		goto end;
	}
	for (int i = 0 ; i < inputCtx[0]->nb_streams ; i++) {		
		AVCodecContext* outputCodecCtx;
		AVStream* stream = avformat_new_stream(outputCtx ,NULL);
		if (!stream) {
			av_log(NULL ,AV_LOG_ERROR ,"Ouput File %s Open Stream#%u Failure\n" ,output ,i);
			
			goto end;
		}
		AVCodec* codec = avcodec_find_encoder(inputCtx[0]->streams[i]->codecpar->codec_id);
		if (!codec) {	
			av_log(NULL ,AV_LOG_ERROR ,"Find Codec %s Open Stream#%u Failure\n" ,output ,i);
			goto end;
		}
		outputCodecCtx = avcodec_alloc_context3(codec);
		if (!outputCodecCtx) {	
			av_log(NULL ,AV_LOG_ERROR ,"Find Codec Context %s Open Stream#%u Failure\n" ,output ,i);
			ret = AVERROR(ENOMEM);
			goto end;
		}
		ret = avcodec_parameters_from_context(stream->codecpar ,outputCodecCtx);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Copy Codec Context %s Open Stream#%u Failure\n" ,output ,i);
			ret = AVERROR(ENOMEM);
			goto end;
		}
		if (outputCodecCtx->codec_type == AVMEDIA_TYPE_VIDEO) {
			// Self Define Video Parameters
//			outputCodecCtx->framerate = av_guess_frame_rate(inputCtx[0] ,stream ,NULL);
			outputCodecCtx->gop_size  = 30;
			outputCodecCtx->has_b_frames  = 0;
			outputCodecCtx->max_b_frames  = 0;
			outputCodecCtx->codec_id  = codec->id;	
//			if (!codec->pix_fmts) {
//				outputCodecCtx->pix_fmt  = codec->pix_fmts[0];
//			} else {
				outputCodecCtx->pix_fmt = inputCodecCtx[0]->pix_fmt;
//			}
			outputCodecCtx->time_base = av_inv_q(outputCodecCtx->framerate);
			outputCodecCtx->width  = inputCodecCtx[0]->width;
			outputCodecCtx->height  = inputCodecCtx[0]->height;
			outputCodecCtx->me_subpel_quality = 0;
			outputCodecCtx->trellis = 0;
		} else if (outputCodecCtx->codec_type == AVMEDIA_TYPE_AUDIO) {
			outputCodecCtx->sample_fmt = codec->sample_fmts[i];
			outputCodecCtx->sample_rate = inputCtx[0]->streams[i]->codecpar->sample_rate;
			outputCodecCtx->channel_layout = inputCtx[0]->streams[i]->codecpar->channel_layout;
			outputCodecCtx->channels = av_get_channel_layout_nb_channels(inputCtx[0]->streams[i]->codecpar->channel_layout);
			outputCodecCtx->time_base = (AVRational){1 ,outputCodecCtx->sample_rate};
		} else {
			av_log(NULL ,AV_LOG_ERROR ,"Subtitle No Suppurt\n");
			goto end;
		}
		outputCtx -> flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
		ret = avcodec_open2(outputCodecCtx ,codec ,NULL);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Open Codec Encoder %s Open Stream#%u Failure\n" ,output ,i);
			ret = AVERROR(ENOMEM);
			goto end;
		}

		ret = avcodec_parameters_from_context(stream->codecpar ,outputCodecCtx);
		if (ret < 0) {
			av_log(NULL ,AV_LOG_ERROR ,"Copy Codec Context %s Open Stream#%u Failure\n" ,output ,i);
			ret = AVERROR(ENOMEM);
			goto end;
		}
	}	
	ret = avio_open(&outputCtx->pb ,output ,AVIO_FLAG_READ_WRITE);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Ouput File %s Open Stream Failure\n" ,output);
		goto end;
	}
	ret = avformat_write_header(outputCtx ,NULL);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Write Header\n");
		goto end;
	}
ret = avformat_write_header(outputCtx ,NULL);
	if (ret < 0){
		av_log(NULL ,AV_LOG_ERROR ,"Write Header\n");
		goto end;
	}
	fprintf(stderr ,"\nFRAME OVERLAY SUCCESS\n");
	return 0;
end:
	fprintf(stderr ,"\nFRAME OVERLAY FAILURE\n");
	return -1;
}
