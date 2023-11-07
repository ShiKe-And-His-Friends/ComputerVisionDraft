#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src, yuv;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	//×ª»»²ÊÉ«
	cvtColor(src ,yuv ,COLOR_BGR2YCrCb);

	vector<vector<Point>> regions;
	vector<cv::Rect> mserBbox;

	Ptr<MSER> mserExtractor = MSER::create();
	mserExtractor->detectRegions(yuv ,regions, mserBbox);

	for (int i = 0; i < regions.size(); i++) {
		ellipse(src, fitEllipse(regions[i]), Scalar(255,0,0));
	}

	//ÏÔÊ¾
	namedWindow("MSCR", WINDOW_AUTOSIZE);
	imshow("MSCR", src);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//MSCR.png", src);

	return 0;
}