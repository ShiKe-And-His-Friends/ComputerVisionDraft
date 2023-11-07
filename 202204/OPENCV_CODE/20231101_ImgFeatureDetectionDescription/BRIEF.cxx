#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat img = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.png", -1);

	if (!img.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	

	//ÏÔÊ¾
	namedWindow("BRIEF", WINDOW_AUTOSIZE);
	imshow("BRIEF", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//BRIEF.png", output_img);

	return 0;
}