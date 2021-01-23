extern "C" {
	#include <libavformat/avformat.h>
	#include <libavformat/avio.h>
	#include <libavutil/file.h>
}

using namespace std;

struct dataInfo {
	uint8_t* data;
	int length;
};

static int read_packet_info(void* opaque, uint8_t* buf, int buf_size) {
	struct dataInfo * datainfo = (struct dataInfo*)opaque;
	buf_size = FFMIN(datainfo->length , buf_size);
	if (!buf_size) {
		return AVERROR_EOF;
	}
	printf("\n file is %p , length is %d .Packet is %d \n" ,datainfo->data, datainfo->length ,buf_size);
	memcpy(buf,datainfo->data ,buf_size);
	datainfo->data += buf_size;
	datainfo->length -= buf_size;
	return buf_size;
}

int main(int argc, char* argv[]) {

	int ret = -1;
	char* filename = NULL;
	size_t buffer_size = 2048 ,file_size = 0;
	uint8_t* data = NULL ,*file_data = NULL;
	struct dataInfo dataInfo = {0};
	AVFormatContext* avFormatCtx;
	AVIOContext* avIoCtx;

	if (argc != 2) {
		printf("\n %s input_file \n" ,argv[0]);
		return -1;
	}
	filename = argv[1];
	ret = av_file_map(filename ,&file_data ,&file_size ,0 ,NULL);
	
	printf("\n size is %zd\n" , file_size);

	dataInfo.data = file_data;
	dataInfo.length = file_size;

	data = (uint8_t*)av_malloc(buffer_size);
	if (!data) {
		printf("\n initlize size error.\n");
		return 1;
	}

	avIoCtx = avio_alloc_context(data , buffer_size ,0
					,&dataInfo, &read_packet_info ,NULL ,NULL);
	if (!avIoCtx) {
		printf("\n av io context error.\n");
		return 1;
	} else {
		printf("\n av io context success.\n");
	}
	
	avFormatCtx = avformat_alloc_context();
	if (!avFormatCtx) {
		printf("\n av context error.\n");
		goto clean;
	}
	avFormatCtx->pb = avIoCtx;

	ret = avformat_open_input(&avFormatCtx ,NULL ,NULL ,NULL); 

	if (!ret) {
		printf("\n av io open success.\n");
	}else {
		printf("\n av io open failure.\n");
	}

	ret = avformat_find_stream_info(avFormatCtx,NULL);
	if (ret < 0) {
		printf("\n av stream info open failure.\n");
		goto clean;
	}
	av_dump_format(avFormatCtx ,0 ,filename ,0);

	return 0;

clean:
	avformat_close_input(&avFormatCtx);
	av_freep(avIoCtx->buffer);
	av_file_unmap(file_data ,file_size);
	if (ret < 0) {
		// av_err2str(ret)
		char info[256] = {0};
		av_strerror(ret ,info ,sizeof(info));
		printf("Error occurred %s \n" ,info);
	}
	return 1;
}