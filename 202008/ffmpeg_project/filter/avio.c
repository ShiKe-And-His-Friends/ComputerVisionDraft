#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/file.h>

struct buffer_data{
	uint8_t *ptr;
	size_t size;
};

static int read_packet(void *opaque ,uint8_t *buf ,int buf_size){
	struct buffer_data * bd = (struct buffer_data *)opaque;
	buf_size - FFMIN(buf_size ,bd->size);
	if (!buf_size) {
		return AVERROR_EOF;
	}
	printf("ptr: %p size: %zu \n" ,bd->ptr ,bd->size);
	/** copy internal buffer data to buf **/
	memcpy(buf ,bd->ptr ,buf_size);
	bd->ptr += buf_size;
	bd->size -= buf_size;

	return buf_size;
}

int main (int argc ,char *argv[]) {

	AVFormatContext *fmt_ctx = NULL;
	AVIOContext *avio_ctx = NULL;
	uint8_t *buffer = NULL ,*avio_ctx_buffer = NULL;
	size_t buffer_size ,avio_buffer_size = 4096;
	char *input_filename = NULL;
	int ret = 0;
	struct buffer_data bd = {0};

	if (argc != 2) {
		fprintf(stderr, "usage read input.\n");
		return 1;
	}
	input_filename =argv[1];
	/** surlp file content into buffer **/
	ret = av_file_map(input_filename ,&buffer ,&buffer_size ,0 ,NULL);
	if (ret < 0) {
		goto end;
	}
	/** file opaque structure used by the AVIOContext read callback **/
	bd.ptr = buffer;
	bd.size = buffer_size;
	if (!(fmt_ctx = avformat_alloc_context())) {
		ret = AVERROR(ENOMEM);
		goto end;
	}
	avio_ctx_buffer = av_malloc(avio_buffer_size);
	if(!avio_ctx_buffer) {
		ret = AVERROR(ENOMEM);
		goto end;
	}
	avio_ctx = avio_alloc_context(avio_ctx_buffer ,avio_buffer_size
			,0 ,&bd ,&read_packet ,NULL ,NULL);
	if (!avio_ctx){
		ret = AVERROR(ENOMEM);
		goto end;
	}
	fmt_ctx->pb = avio_ctx;
	ret = avformat_open_input(&fmt_ctx ,NULL ,NULL ,NULL);
	if (ret < 0) {
		fprintf(stderr ,"can not open.\n");
		goto end;
	}

	ret = avformat_find_stream_info(fmt_ctx ,NULL);
	if (ret < 0) {
		fprintf(stderr, "can not find stream.\n");
		goto end;
	}
	av_dump_format(fmt_ctx ,0 ,input_filename ,0);

end:	
	avformat_close_input(&fmt_ctx);
	/** note: internal buffer can not changed  **/
	if (avio_ctx) {
		av_freep(&avio_ctx->buffer);
		av_freep(&avio_ctx);
	}
	av_file_unmap(buffer ,buffer_size);
	if (ret < 0){
		fprintf(stderr ,"Error occurred %s \n" ,av_err2str(ret));
		return 1;
	}
	return 0;
}
