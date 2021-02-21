#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>

static int get_format_from_sample_fmt(const char **fmt ,enum AVSampleFormat sample_fmt) {
	int i;
	struct sample_fmt_entry{
		enum AVSampleFormat sample_fmt;
		const char *fmt_be ,*fmt_le;
	} sample_fmt_entries[] = {
		{AV_SAMPLE_FMT_U8 ,"u8" ,"u8"},
		{AV_SAMPLE_FMT_S16 ,"s16be" ,"s16le"},
		{AV_SAMPLE_FMT_S32 ,"s32be" ,"s32le"},
		{AV_SAMPLE_FMT_FLT ,"f32be" ,"f32le"},
		{AV_SAMPLE_FMT_DBL ,"f64be" ,"f64le"},
	};
	*fmt = NULL;

	for (i = 0 ; i < FF_ARRAY_ELEMS(sample_fmt_entries) ;i++) {
		struct sample_fmt_entry *entry = &sample_fmt_entries[i];
		if (sample_fmt == entry->sample_fmt) {
			*fmt = AV_NE(entry->fmt_be ,entry->fmt_le);
			return 0;
		}
	}
	fprintf(stderr ,"Sample format %s not supported as output format.\n" ,av_get_sample_fmt_name(sample_fmt));
	return AVERROR(EINVAL);
}

/** Fill dst buffer with nb_samples ,generated starting format **/
static void fill_samples(double *dst ,int nb_samples ,int nb_channels ,int sample_rate ,double *t) {
	int i ,j;
	double tincr = 1.0 / sample_rate ,*dstp = dst;
	const double c = 2 * M_PI * 440.0;

	/** generate sin tone with 440Hz frequency and dumplicated channels **/
	for (i = 0 ; i < nb_channels ; i++) {
		*dstp =  sin(c * *t);
		for (j = 1 ; j < nb_channels ; j++) {
			dstp[j] = dstp[0];
		}
		dstp += nb_channels;
		*t += tincr;
	}
}

int main (int argc ,char **argv) {
	int64_t src_ch_layout = AV_CH_LAYOUT_STEREO ,dst_ch_layout = AV_CH_LAYOUT_SURROUND;
	int src_rate = 48000 ,dst_rate = 44100;
	uint8_t **src_data = NULL ,**dst_data = NULL;
	int src_nb_channels = 0,dst_nb_channels = 0;
	int src_linesize ,dst_linesize;
	int src_nb_samples = 1024 ,dst_nb_samples ,max_dst_nb_samples;
	enum AVSampleFormat src_sample_fmt = AV_SAMPLE_FMT_DBL ,dst_sample_fmt = AV_SAMPLE_FMT_S16;
	const char *dst_filename = NULL;
	FILE *dst_file;
	int dst_bufsize;
	const char *fmt;
	struct SwrContext *swr_ctx;
	double t;
	int ret;

	if (argc != 2) {
		fprintf(stderr ,"Usage: %s output_file \n API example program to show how to resample an audio stream with libswresample.\n This program generates a series of audio frams ,resample them to a specified ouput foramt and rate and saves them to an output file named output_file.\n" ,argv[0]);
		exit(1);
	}
	/** create resampler context **/
	swr_ctx = swr_alloc();
	if (!swr_ctx) {
		fprintf(stderr ,"Could not allocate resampler contxt.\n");
		ret = AVERROR(ENOMEM);
		goto end;
	}
	/** set options **/
	av_opt_set_int(swr_ctx ,"in_channel_layout" ,src_ch_layout ,0);
	av_opt_set_int(swr_ctx ,"in_sample_rate" ,src_rate ,0);
	av_opt_set_sample_fmt(swr_ctx ,"in_sample_fmt" ,src_sample_fmt ,0);
	av_opt_set_int(swr_ctx ,"out_channel_layout" ,dst_ch_layout ,0);
	av_opt_set_int(swr_ctx ,"out_sample_rate" ,dst_rate ,0);
	av_opt_set_sample_fmt(swr_ctx ,"out_sample_fmt" ,dst_sample_fmt ,0);
	/** initialize the resampling context **/
	if ((ret = swr_init(swr_ctx)) < 0) {
		fprintf(stderr ,"Failed to initialize the resampling context.\n");
		goto end;
	}
	/** allocate source and destination samples buffers **/
	src_nb_channels = av_get_channel_layout_nb_channels(src_ch_layout);
	ret = av_samples_alloc_array_and_samples(&src_data ,&src_linesize ,dst_nb_channels ,src_nb_samples ,src_sample_fmt ,0);
	if (ret < 0) {
		fprintf(stderr ,"Could not allocate source samples.\n");
		goto end;
	}

	/** compute the number of converted samples: buffering is avoided ensuring that the output buffer will contaion at leasr all the converted input samples. **/
	max_dst_nb_samples = dst_nb_samples = av_rescale_rnd(src_nb_samples ,dst_rate ,src_rate ,AV_ROUND_UP);
	/** buffer is going to be directly written to a rawaudio file ,no alignment **/
	dst_nb_channels = av_get_channel_layout_nb_channels(dst_ch_layout);
	ret = av_samples_alloc_array_and_samples(&dst_data ,&dst_linesize ,dst_nb_channels ,dst_nb_samples ,dst_sample_fmt ,0);
	if (ret < 0) {
		fprintf(stderr ,"Could not allocate destination samples.\n");
		goto end;
	}

	t = 0;
	do {
		/** gengerate synthetic audio **/
		fill_samples((double *)src_data[0] ,src_nb_samples ,src_nb_channels ,src_rate ,&t);
		/** compute destination number of samples **/
		dst_nb_samples = av_rescale_rnd(swr_get_delay(swr_ctx ,src_rate) + src_nb_samples ,dst_rate ,src_rate ,AV_ROUND_UP);
		if (dst_nb_samples > max_dst_nb_samples) {
			av_freep(&dst_data[0]);
			ret = av_samples_alloc(dst_data ,&dst_linesize ,dst_nb_channels ,dst_nb_samples ,dst_sample_fmt ,1);
			if (ret < 0) {
				break;
			}
			max_dst_nb_samples = dst_nb_samples;
		}
		/** conert to destination format**/
		ret = swr_convert(swr_ctx ,dst_data ,dst_nb_samples ,(const uint8_t **)src_data ,src_nb_samples);
		if (ret < 0) {
			fprintf(stderr ,"Error while converting\n");
			goto end;
		}
		dst_bufsize = av_samples_get_buffer_size(&dst_linesize ,dst_nb_channels ,ret ,dst_sample_fmt ,1);
		if (dst_bufsize < 0) {
			fprintf(stderr ,"Could not get sample buffer size.\n");
			goto end;
		}
		printf("t:%f in:%d out:%d \n" ,t ,src_nb_samples ,ret);
		fwrite(dst_data[0] ,1 ,dst_bufsize ,dst_file);
	} while (t < 10);

	if ((ret = get_format_from_sample_fmt(&fmt ,dst_sample_fmt)) < 0) {
		goto end;
	}
	fprintf(stderr ,"Resampling succeeded. Play the output file with the command:\n ffplay -f %s -channel_layout %"PRId64" -channels %d -ar %d %s\n " ,fmt ,dst_ch_layout ,dst_nb_channels ,dst_rate ,dst_filename);

end:
	fclose(dst_file);
	if (src_data) {
		av_freep(&src_data[0]);
	}
	av_freep(&src_data);
	if (dst_data) {
		av_freep(&dst_data[0]);
	}
	av_freep(&dst_data);
	swr_free(&swr_ctx);
	return ret < 0;
}

/**
 * 1. struct
 * 1.1 typedef SwrContext { }SwrContext;
 * 
 * 2. function
 * 2.1 struct SwrContext *swr_alloc(void);
 * 2.2 int av_opt_set_int(void *obj ,const char *name ,const char *val ,int search_flags);
 * 2.3 int av_opt_set_sample_fmt(void *obj ,const char *name ,enum AVSampleFormat fmt ,int search_flags);
 * 2.4 int swr_init(struct SwrContext *s);
 * 2.5 int av_get_channel_layout_nb_channels(uint64_t channel_layout);
 * 2.6 int av_samples_alloc_array_and_samples(uint8_t ***audio_data ,int **linesize ,int nb_channels , int nb_samples ,enum AVSampleFormat sample_fmt ,int align);
 * 2.7 int64_t av_rescale_rnd(int64_t a ,int64_t b ,int64_t c ,enum AVRounding rnd) av_const;
 * 2.8 static void fill_samples(double *dst ,int nb_samples ,int nb_channels ,int sample_rate ,double *t);
 * 2.9 int av_samples_alloc(uint8_t **audio_data ,int *linesize ,int nb_channels);
 * 2.10 int swr_convert(struct SwrContext *s ,uint8_t **out ,int out_count ,const uint8_t **in ,int in_count);
 * 2.11 int av_samples_get_buffer_size(int *linesize ,int nb_channels ,int nb_samples ,enum AvSampleFormat sample_fmt ,int align);
 * 2.12 static int get_format_from_sample_fmt(const char **fmt ,enum AVSampleFormat sample_fmt);
 * 2.13 void av_freep(void *ptr);
 * 2.14 void swr_free(struct SwrContext **s);
 *
**/
