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
	Mat img2 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img2.png", -1);

	if (!img1.data || !img2.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	vector<KeyPoint> key_points1, key_points2;
	Ptr<Feature2D>detector = cv::xfeatures2d::StarDetector::create(); //特征点检测方法
	detector.detect(img1 ,key_points1);
	detector.detect(img2, key_points2);

	Mat descriptors1, descriptors2;
	BriefDescriptorExtractor brief; //BRIEF方法
	brief.compute(img1 ,key_points1 ,descriptors1);
	brief.compute(img2, key_points2, descriptors2);

	BruteForceMatcher<Hamming> matcher;
	vector<DMatch> matches;
	matcher.match(descriptors1 ,descriptors2 ,matches);

	std::nth_element(matches,begin,
		matches.begin(),
		matches.end());

	matches.erase(matches.begin() +8 ,matches.end());

	//显示
	namedWindow("BRIEF_matches", WINDOW_AUTOSIZE);
	Mat img_matches;
	drawMatches(
		img1 ,key_points1,
		img2 ,key_points2,
		matches,
		img_matches,
		Scalar(255 ,255 ,255));
	imshow("BRIEF_matches", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//BRIEF_matches.png", output_img);

	return 0;
}