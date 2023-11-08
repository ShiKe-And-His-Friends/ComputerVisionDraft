
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat img1 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.png", -1);

	if (!img1.data ) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	Ptr<Feature2D> orb = ORB::create(300); //300个特征点

	vector<KeyPoint> key_points;

	Mat descriptors, mascara;
	Mat output_img;

	//选择角点
	orb->detect(img1, key_points);

	//计算描述子
	orb->compute(img1, key_points , descriptors);

	drawKeypoints(
		img1 ,
		key_points,
		output_img,
		Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


	//显示
	namedWindow("ORB", WINDOW_AUTOSIZE);
	imshow("ORB", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//ORB.png", output_img);

	return 0;
}