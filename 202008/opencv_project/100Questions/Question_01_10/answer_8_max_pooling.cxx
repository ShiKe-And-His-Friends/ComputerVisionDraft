#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>

// max pooling
cv::Mat max_pooling(cv::Mat img) {
	// get height and width
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC3);

	int r = 8;
	double v = 0;

	for (int y = 0; y < height; y += r) {
		for (int x = 0; x < width;x += r) {
			for (int c = 0; c < channels; c++) {
				// calcute average
				v = 0;
				for (int dy = 0; dy < r; dy++) {
					for (int dx = 0; dx < r; dx ++) {
						v = fmax(img.at<cv::Vec3b>(y + dy, x + dx)[c] ,v);
					}
				}
				
				// pooling
				for (int dy = 0; dy < r; dy++) {
					for (int dx = 0; dx < r; dx++) {
						out.at<cv::Vec3b>(y + dy, x + dx)[c] = (uchar)v;
					}
				}
			}
		}
	
	}
	return out;
}

int main(int argc ,char **argv) {

	// read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori.jpg", cv::IMREAD_COLOR);

	// max pooling
	cv::Mat out = max_pooling(img);

	cv::imshow("sample",out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}