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

	//FAST的阈值设置60 ，金字塔采用4个层组，采样模板的基准尺寸是1.0
	Ptr<Feature2D> brisk_detect = BRISK::create();
	vector<KeyPoint> key_points;

	Mat descriptors, mascara;
	Mat output_img;

	brisk_detect->detectAndCompute(img ,Mat() ,key_points ,descriptors);
	drawKeypoints(
		img,
		key_points,
		output_img,
		Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//显示
	namedWindow("BRISK", WINDOW_AUTOSIZE);
	imshow("BRISK", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//BRISK.png", output_img);

	return 0;
}