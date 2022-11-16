#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>


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

// Gray -> Binary
cv::Mat Binarize_Otsu(cv::Mat gray) {
	int width = gray.cols;
	int height = gray.rows;

	// prepare output
	cv::Mat out = cv::Mat::zeros(height ,width ,CV_8UC1);

	// determine threshold
	double w0 = 0, w1 = 0;
	double m0 = 0, m1 = 0;
	double max_sb = 0, sb = 0;
	int th = 0;
	int val;

	// Get threshold
	for (int t = 0; t < 255; t++) {
		w0 = 0;
		w1 = 0;
		m0 = 0;
		m1 = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				val = (int)(gray.at<uchar>(y, x));
				if (val < t) {
					w0++;
					m0 += val;
				}
				else {
					w1++;
					m1 += val;
				}
			}
		}
		m0 /= w0;
		m1 /= w1;
		w0 /= (height * width);
		w1 /= (height * width);
		sb = w0 * w1 * pow((m0 - m1) ,2);
		if (sb > max_sb) {
			max_sb = sb;
			th = t;
		}
	}
	std::cout << "threshold: " << th << std::endl;

	// each y ,x
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Binarize
			if (gray.at<uchar>(y ,x) > th) {
				out.at<uchar>(y, x) = 255;
			}
			else {
				out.at<uchar>(y, x) = 0;
			}
		}
	}
	return out;
}

int main(int argc ,char **argv) {

	// read image
	cv::Mat img = cv::imread("D://computerVisionAll//ComputerVisionDraft//202008//opencv_project//100Questions//assert//imori.jpg", cv::IMREAD_COLOR);

	// BGR -> Gray
	cv::Mat gray = BGR2RAY(img);

	// Gray -> Binary
	cv::Mat out = Binarize_Otsu(gray);

	cv::imshow("sample",out);
	cv::waitKey(6000);
	cv::destroyAllWindows();

	return 0;
}