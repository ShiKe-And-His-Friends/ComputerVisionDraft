#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

#define M_PI_VALUE  3.1415926

// median filter
cv::Mat median_filter(cv::Mat img ,int kernel_size) {
	// get height and width
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	// prepare output
	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC3);

	// prepare kernel
	int pad = floor(kernel_size / 2);

	// filtering
	double v = 0;
	int* vs = new int[kernel_size * kernel_size];
	int count = 0;

	for (int y = 0; y < height; y ++) {
		for (int x = 0; x < width;x ++) {
			for (int c = 0; c < channels; c++) {
				// calcute convolution
				v = 0;
				count = 0;
				for (int i = 0; i < kernel_size * kernel_size; i++) {
					vs[i] = 999;
				}

				// get neighbor pixels
				for (int dy = -pad; dy < pad + 1; dy++) {
					for (int dx = -pad; dx < pad + 1; dx ++) {
						if (((x + dx) >= 0) && ((y + dy) >=0) && (y + dy < height) && (x + dx < width) ) {
							vs[count++] = (int)img.at<cv::Vec3b>(y+dy ,x+dx)[c];
						}
					}
				}

				// get and assign median
				std::sort(vs ,vs + (kernel_size * kernel_size));
				out.at<cv::Vec3b>(y, x)[c] = (uchar)vs[int (floor(count / 2)) + 1];
			}
		}
	
	}

	delete[] vs;

	return out;
}

int main(int argc ,char **argv) {

	// read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori_noise.jpg", cv::IMREAD_COLOR);

	// median filter
	cv::Mat out = median_filter(img ,3);

	cv::imshow("sample",out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}