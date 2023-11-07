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

	Mat img1;
	//转换灰色
	cvtColor(img, img1, COLOR_BGR2GRAY);

	Ptr<FastFeatureDetector> star = FastFeatureDetector::create();
	vector<KeyPoint> key_points;

	Mat output_img;

	star->detect(img1,key_points);


	//绘制特征点
	drawKeypoints(
		img,
		key_points,
		output_img,
		Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS
		);

	//显示
	namedWindow("STAR", WINDOW_AUTOSIZE);
	imshow("STAR", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//STAR.png", output_img);

	return 0;
}