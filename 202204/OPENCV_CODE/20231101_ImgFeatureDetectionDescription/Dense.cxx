
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

	if (!img1.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	cvtColor(img1 ,img1 ,COLOR_BGR2GRAY);
	Ptr<DenseFeatureDetector> detector = cv::Feature2D::DenseFeatureDetector::create();
	vector<KeyPoint> key_points;
	Mat output_img;

	detector->detect(img1 ,key_points ,output_img);

	drawKeypoints(
		img1,
		key_points,
		output_img,
		Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//œ‘ æ
	namedWindow("Dense", WINDOW_AUTOSIZE);
	imshow("Dense", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//Dense.png", output_img);

	return 0;
}