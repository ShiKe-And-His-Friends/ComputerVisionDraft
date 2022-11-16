#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
/***

	gray = 0.2126 R + 0.7152 G + 0.0722 B

***/
cv::Mat BGR2RAY(cv::Mat img) {
	// get height and width
	int width = img.cols;
	int height = img.rows;

	// prepare output
	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC1);
	
	// each y x
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			out.at<uchar>(y, x) = 0.2126 * (float)img.at<cv::Vec3b>(y, x)[2]
				+ 0.7152 * (float)img.at<cv::Vec3b>(y, x)[1]
				+ 0.0722 * (float)img.at<cv::Vec3b>(y, x)[0];
		}
	}
	return out;
}

int main(int argc ,char **argv) {

	// read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori.jpg", cv::IMREAD_COLOR);

	// BGR -> Gray
	cv::Mat out = BGR2RAY(img);

	cv::imshow("sample",out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}