#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>

int main (int argc ,char **argv) {
	int ret;
	const char *input_file_name ,*output_file_name;
	AVFormatContext **ifmt_ctx;
	if (argc < 3) {
		printf("Usage: %s intput output\n" ,argv[0]);
		return 1;
	}
	input_file_name = argv[1];
	output_file_name = argv[2];
	if ((ret = avformat_open_input(ifmt_ctx ,input_file_name ,0 ,0)) < 0) {
		fprintf(stderr ,"Could not open input file %s," ,input_file_name);
		goto end;
	}
	printf("shikeDebug ...");
	
end:
	avformat_close_input(&ifmt_ctx);
	
}