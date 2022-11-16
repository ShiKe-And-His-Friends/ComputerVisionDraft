#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cv::Mat channel_swap(cv::Mat img) {
	//get height and width
	int width = img.cols;
	int height = img.rows;

	// prepare output
	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC3);
	
	//each y,x
	for (int y = 0; y < height; y ++) {
		for (int x = 0; x < width; x++) {
			// R->B
			out.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[2];
			// B->R
			out.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[0];
			// G->G
			out.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1];
		}
	}
	return out;
}

int main(int argc ,char **argv) {

	//read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori.jpg" ,cv::IMREAD_COLOR);

	// channel swap
	cv::Mat out = channel_swap(img);

	cv::imshow("out" ,out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}