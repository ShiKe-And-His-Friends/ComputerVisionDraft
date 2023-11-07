#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/Xfeatures2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.png", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	SURF surf(3500.); //设置行列式阈值为3000

	vecotr<KeyPoint> key_points;

	Mat descriptors, mascara;
	Mat output_img;

	surf(img ,mascara ,key_points ,descriptors);

	drawKeypoints(
		img,
		key_points,
		output_img,
		Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS
		);

	//显示
	namedWindow("SURF", WINDOW_AUTOSIZE);
	imshow("SURF", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SURF.png", output_img);

	return 0;
}