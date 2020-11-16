#include <stdio.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>

int main(int  argc ,char **argv) {
	AVFormatContext *fmt_ctx = NULL;
	AVDictionaryEntry *tag = NULL;
	int ret;
	if (argc != 2) {
		printf("Usage : %s <input file> \n Example program to demonstrate the use of the libavformat metadata API. \n" ,argv[0]);
		return 1;
	}
	if ((ret = avformat_open_input(&fmt_ctx ,argv[1] ,NULL ,NULL))) {
		return 1;
	}
	while ((tag = av_dict_get(fmt_ctx->metadata ,"" ,tag ,AV_DICT_IGNORE_SUFFIX))) {
		printf("%s = %s \n" ,tag->key ,tag->value);
	}
	avformat_close_input(&fmt_ctx);
	return 0;
}
