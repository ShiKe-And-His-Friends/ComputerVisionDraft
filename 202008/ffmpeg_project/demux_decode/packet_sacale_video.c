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

}
