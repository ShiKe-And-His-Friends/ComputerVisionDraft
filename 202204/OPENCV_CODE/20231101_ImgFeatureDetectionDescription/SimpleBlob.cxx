
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

	SimpleBlobDetector::Params params;
	params.minThreshold = 40;
	params.maxThreshold = 160;
	params.thresholdStep = 5;
	params.minArea = 100;
	params.minConvexity = 0.05f;
	params.minInertiaRatio = 0.05f;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> key_points;
	detector->detect(img1 ,key_points);

	//œ‘ æ
	namedWindow("SimpleBlob", WINDOW_AUTOSIZE);
	Mat output_img;
	drawKeypoints(
		img1,
		key_points,
		output_img,
		Scalar(0,0,255),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SimpleBlob", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SimpleBlob.png", output_img);

	return 0;
}