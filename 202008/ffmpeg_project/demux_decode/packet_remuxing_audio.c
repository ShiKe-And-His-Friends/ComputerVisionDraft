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
	}
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

/** Fill dst **/
