#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using std::cout;

const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;
Mat src1;
Mat src2;
Mat dst;

static void on_trackbar(int ,void*) {
	alpha = (double)alpha_slider/alpha_slider_max;
	beta = (1.0 - alpha);
	addWeighted(src1 ,alpha ,src2 ,beta ,0.0 ,dst);
	imshow("Linear Blend" ,dst);
}

int main(void) {
	/*
	  1. add global envrionment:
	  export OPENCV_SAMPLES_DATA_PATH=/home/shike/Documents/opencv-4.2.0/samples/data
	  2. add class comiple environment
	cv::samples::addSamplesDataSearchPath("/home/shike/Documents/opencv-4.2.0/samples/data");
	*/
	src1 = imread(samples::findFile("LinuxLogo.jpg"));
	src2 = imread(samples::findFile("WindowsLogo.jpg"));
	if (src1.empty()) {
		cout << "Error loading src1 \n";
		return -1;
	}
	if (src2.empty()) {
		cout << "Error loading src2 \n";
		return -1;
	}
	alpha_slider = 0;
	namedWindow("Linear Blend" ,WINDOW_AUTOSIZE);
	char TrackbarName[50];
	sprintf(TrackbarName ,"Alpha x %d" ,alpha_slider_max);
	createTrackbar(TrackbarName ,"Linear Blend" ,&alpha_slider ,alpha_slider_max ,on_trackbar);
	on_trackbar(alpha_slider ,0);

	waitKey(0);
	return 0;	
}
