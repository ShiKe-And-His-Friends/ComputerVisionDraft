#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src, src_gray;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	//给出goodFeatureToTrack函数所需参数
	vector<Point2f> corners;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int maxCorners = 150;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	
	Mat copy;
	copy = src.clone();

	// 角点检测
	goodFeaturesToTrack(
		src_gray,
		corners,
		maxCorners,
		qualityLevel,
		minDistance,
		Mat(),
		blockSize,
		useHarrisDetector,
		k);

	//画出角点位置
	for (int i = 0; i < corners.size(); i++) {
		circle(copy ,corners[i] ,5 ,Scalar(0,0,255) ,-1 ,8 ,0);
	}

	//显示
	namedWindow("Shi_Tomasi", WINDOW_AUTOSIZE);
	imshow("Shi_Tomasi", copy);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//Shi_Tomasi.png", copy);

	return 0;
}