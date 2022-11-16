#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

// decreate color
cv::Mat decreate_color(cv::Mat img) {
	// get height and width
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC3);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width;x ++) {
			for (int c = 0; c < channels; c++) {
				out.at<cv::Vec3b>(y, x)[c] = (uchar)(floor((double)img.at<cv::Vec3b>(y ,x)[c] / 64) * 64 + 32);
			}
		}
	
	}
	return out;
}

int main(int argc ,char **argv) {

	// read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori.jpg", cv::IMREAD_COLOR);

	// decreate color
	cv::Mat out = decreate_color(img);

	cv::imshow("sample",out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}