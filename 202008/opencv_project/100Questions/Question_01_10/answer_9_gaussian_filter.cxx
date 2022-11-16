#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

#define M_PI_VALUE  3.1415926

// gaussian filter
cv::Mat gaussian_filter(cv::Mat img ,double sigma ,int kernel_size) {
	// get height and width
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	// prepare output
	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC3);

	// prepare kernel
	int pad = floor(kernel_size / 2);
	int _x = 0, _y = 0;
	double kernel_sum = 0;

	// get gasussian kernel
	float** kernel = new float*[kernel_size];
	for (int i = 0; i < kernel_size; i++) {
		kernel[i] = new float[kernel_size];
	}

	for (int y = 0; y < kernel_size; y++) {
		for (int x = 0; x < kernel_size; x++) {
			_y = y - pad;
			_x = x - pad;
			kernel[y][x] = 1 / (2 * M_PI_VALUE * sigma * sigma) * exp( - (_x * _x + _y * _y) / (2 * sigma *sigma));
			kernel_sum += kernel[y][x];
		}
	}

	for (int y = 0; y < kernel_size; y++) {
		for (int x = 0; x < kernel_size; x++) {
			kernel[y][x] /= kernel_sum;
			std::cout << kernel[y][x] << " ";
		}
	}

	// filtering
	double v = 0;

	for (int y = 0; y < height; y ++) {
		for (int x = 0; x < width;x ++) {
			for (int c = 0; c < channels; c++) {
				// calcute convolution
				v = 0;
				for (int dy = -pad; dy < pad + 1; dy++) {
					for (int dx = -pad; dx < pad + 1; dx ++) {
						if (((x + dx) >= 0) && ((y + dy) >=0) && (y + dy < height) && (x + dx < width) ) {
							v += (double)img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
						}
					}
				}
				out.at<cv::Vec3b>(y, x)[c] = v;
			}
		}
	
	}

	//delete[] kernel;

	return out;
}

int main(int argc ,char **argv) {

	// read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori_noise.jpg", cv::IMREAD_COLOR);

	// gaussian filter
	cv::Mat out = gaussian_filter(img ,1.3 ,3);

	cv::imshow("sample",out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}