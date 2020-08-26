/**
 * Sample in FFmpeg4.1
 * **/
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>

static void fill_yuv_image(uint8_t *data[4] ,int linesize[4] 
		,int width ,int height ,int frame_index) {
	int x,y;
	/*Y*/
	for (y = 0 ;y < height ; y++) {
		for (x = 0 ;x < width ;x++ ) {
			data[0][y * linesize[0] + x] = x + y + frame_index * 3;
		}
	}

	/*Cb Cr */
	for (y = 0 ;y < height / 2 ;y++){
		for(x = 0;x < width / 2 ;x++){
			data[1][y * linesize[1] + x] = 128 + y + frame_index * 2;
			data[2][y * linesize[2] + x] = 64 + x + frame_index * 5;
		}
	}
}

int main(int argc ,char **argv) {
	uint8_t *src_data[4], *dst_data[4];
	int src_linesize[4] ,dst_linesize[4];
	int src_w = 320 ,src_h = 240 ,dst_w ,dst_h;
	enum AVPixelFormat src_pix_fmt = AV_PIX_FMT_YUV420P
		,dst_pix_fmt = AV_PIX_FMT_RGB24;
	const char *dst_size = NULL;
	const char *dst_filename = NULL;
	FILE *dst_file;
	int dst_buffersize;
	struct SwsContext *sws_ctx;
	int i,ret;

	if (argc != 3) {
		fprintf(stderr ,"Usage: %s output_file output_size.\n" ,argv[0]);
		exit(1);
	}
	dst_filename = argv[1];
	dst_size = argv[2];
	if (av_parse_video_size(&dst_w ,&dst_h ,dst_size) < 0) {
		fprintf(stderr ,"Invalis size '%s'" ,dst_size);
		exit(1);
	}
	dst_file = fopen(dst_filename ,"wb");
	if(!dst_file) {
		fprintf(stderr ,"open file failure : %s" ,dst_filename);
		exit(1);
	}
	sws_ctx = sws_getContext(src_w ,src_h ,src_pix_fmt,
			dst_w ,dst_h ,dst_pix_fmt ,
			SWS_BILINEAR ,NULL ,NULL ,NULL);
	if (!sws_ctx) {
		fprintf(stderr ,"scale context failure %s %s" ,av_get_pix_fmt_name(src_pix_fmt) ,av_get_pix_fmt_name(dst_pix_fmt));
		exit(1);
	}
	/*allocate source and buffer*/
	if ((ret = av_image_alloc(src_data ,src_linesize 
				,src_w ,src_h ,src_pix_fmt ,16)) < 0) {
		fprintf(stderr ,"can not allocate destination image.\n");
		goto end;
	}
	/* buffer is going to be written rawvideo file, no alignment*/
	if ((ret = av_image_alloc(dst_data ,dst_linesize
				,dst_w ,dst_h ,dst_pix_fmt ,1)) < 0) {
		fprintf(stderr ,"Could not allocate destination image.\n");
		exit(1);
	}
	dst_buffersize = ret;
	for (i = 0 ;i < 100 ; i++) {
		/*generate synthetic video*/
		fill_yuv_image(src_data ,src_linesize ,src_w ,src_h ,i);
		/*convert to destination format*/
		sws_scale(sws_ctx ,(const uint8_t * const*)src_data
				,src_linesize ,0 ,src_h ,dst_data ,dst_linesize);
		/*write scaled image to file*/
		fwrite(dst_data[0] ,1 ,dst_buffersize ,dst_file);
	}
	fprintf(stderr ,"Scaling successed. Command : \n"
			"ffplay -f rawvideo -pix_fmt %s -video_size %dx%d %s\n"
			,av_get_pix_fmt_name(dst_pix_fmt) ,dst_w ,dst_h ,dst_filename);

end:	
	fclose(dst_file);
	av_freep(&src_data[0]);
	av_freep(&dst_data[0]);
	sws_freeContext(sws_ctx);

	return ret < 0;
}
