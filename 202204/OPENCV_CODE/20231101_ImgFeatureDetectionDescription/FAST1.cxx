#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src, gray, color_edge;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	vector<KeyPoint> keyPoints;

	//创建对象，阈值设置55
	Ptr<FeatureDetector> fast = FastFeatureDetector::create(55);
	//特征点检测
	fast->detect(src ,keyPoints);

	//在原图画出特征点
	drawKeypoints(
		src,
		keyPoints,
		src,
		Scalar(0,0,255),
		DrawMatchesFlags::DRAW_OVER_OUTIMG
		);

	//显示
	namedWindow("FAST", WINDOW_AUTOSIZE);
	imshow("FAST", src);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//FAST2.png", src);

	return 0;
}