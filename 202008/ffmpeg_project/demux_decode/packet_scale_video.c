/**
 * Generated on Tue Noev 6 2018 18:10:58 for ffmpeg by 1.8.6
 **/
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>

static void fill_yuv_image(uint8_t *data[4] ,int linesize[4] ,int width ,int height ,int frame_index) {
	int x,y;
	/** Y **/
	for (y = 0 ; y < height ;y++) {
		for (x = 0 ; x < width ; x++) {
			data[0][y * linesize[0] + x] = x + y + frame_index * 3;
		}
	}

	/** Cb and Cr **/
	for (y = 0 ; y < height/2 ; y++) {
		for (x = 0 ; x < width/2 ; x++) {
			data[1][y * linesize[1] + x] = 128 + y + frame_index * 2;
			data[2][y * linesize[2] + x] = 64 + x + frame_index * 5;
		}
	
	}
}

int main(int argc ,char **argv) {
	uint8_t *src_data[4] ,*dst_data[4];
	int  src_linesize[4] ,dst_linesize[4];
	int src_w = 320 ,src_h = 240 ,dst_w ,dst_h;
	enum AVPixelFormat src_pix_fmt = AV_PIX_FMT_YUV420P ,dst_pix_fmt = AV_PIX_FMT_RGB24;
	const char *dst_size = NULL;
	const char *dst_filename = NULL;
	FILE *dst_file;
	int dst_bufsize;
	struct SwsContext *sws_ctx;
	int i,ret;
	if (argc != 3) {
		fprintf(stderr ,"Usage: %s output_file output_size\n API example program to show how to scale an image with libswscale.\n This program generates a serirs of pictures, resacles them to the given output_size and saves them to an output file named output_file\n" ,argv[0]);
		exit(1);
	}
	dst_filename = argv[1];
	dst_size = argv[2];
	if (av_parse_video_size(&dst_w ,&dst_h ,dst_size) < 0) {
		fprintf(stderr ,"Invalid size '%s', must be in the from WxH or a valid size abbreviation\n " ,dst_size);
		exit(1);
	}
	dst_file = fopen(dst_filename ,"wb");
	if (!dst_file) {
		fprintf(stderr ,"Could not open destination file%s\n" ,dst_filename);
		exit(1);
	}
	/** create scaling context **/
	sws_ctx = sws_getContext(src_w ,src_h ,src_pix_fmt ,dst_w ,dst_h ,dst_pix_fmt ,SWS_BILINEAR ,NULL ,NULL ,NULL);
	if (!sws_ctx) {
		fprintf(stderr ,"Impossible to create scale context for the conversion fmt:%s s:%dx%d -->fmt:%s s:%dx%d \n" ,av_get_pix_fmt_name(src_pix_fmt) ,src_w ,src_h ,av_get_pix_fmt_name(dst_pix_fmt) ,dst_w ,dst_h);
		ret = AVERROR(EINVAL);
		goto end;
	}
	/** allocate source and destination image buffers **/
	if ((ret = av_image_alloc(src_data ,src_linesize ,src_w ,src_h ,src_pix_fmt ,16)) < 0) {
		fprintf(stderr ,"Could not allocate source image\n");
		goto end;
	}
	/** buffer is going to be written to rawvideo file, no alignment **/
	if ((ret = av_image_alloc(dst_data ,dst_linesize ,dst_w ,dst_h ,dst_pix_fmt ,1)) < 0) {
		fprintf(stderr ,"Could not allocate destination image\n");
		goto end;
	}
	dst_bufsize = ret;
	for (i = 0 ; i < 100 ; i++) {
		/** generate synthetic video **/
		fill_yuv_image(src_data ,src_linesize ,src_w ,src_h ,i);
		/** covert to destination format **/
		sws_scale(sws_ctx ,(const uint8_t * const *)src_data ,src_linesize ,0 ,src_h ,dst_data ,dst_linesize);
		/** write scaled image to file **/
		fwrite(dst_data[0] ,1 ,dst_bufsize ,dst_file);
	}
	fprintf(stderr ,"Scaling successed. Play the output file with the command:\n ffpaly -f rawvideo -pix_fmt %s -video_size %dx%d %s\n ", av_get_pix_fmt_name(dst_pix_fmt) ,dst_w ,dst_h ,dst_filename);

end:
	fclose(dst_file);
	av_freep(&src_data[0]);
	av_freep(&dst_data[0]);
	sws_freeContext(sws_ctx);

	return ret < 0;
}
/**
 * 1. struct
 * 1.1 (self define) typedef struct OutputStream { }OutputStream;
 * 
 * 2. function
 * 2.1 int av_parse_video_size(int *width_ptr ,int *height_ptr ,const char *str);
 * 2.2 struct SwsContext *sws_getContext(int srcW ,int srcH ,enum AVPixelFormat srcFormat ,int dstW ,int dstH ,enum AVPixelFormat dstFormat ,int flags ,SwsFilter *srcFilter ,SwsFilter *dstFilter ,const double *param);
 * 2.3 int av_image_alloc(uint8_t *pointers[4] ,int linesize[4] ,int w ,int h ,enum AVPixelFormat pix_fmt ,int align);
 * 2.4 int sws_scalesws_scale(struct SwsContext *c ,const uint8_t *const srcSlice[] ,const int srcStride[] , int srcSliceY ,int srcSliceH ,uint8_t *const dst[] ,const int dstStride[]);
 * 2.5 void sws_freeContext(struct SwsContext *swsContext);
 *
**/
