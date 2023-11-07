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

	Mat img1 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.png", -1);
	Mat img2 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img2.png", -1);

	if (!img1.data || !img2.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	SURF surf1(3500.), surf2(3000.); //设置行列式阈值为3000

	vecotr<KeyPoint> key_points1, key_points2;

	Mat descriptors1 , descriptors2, mascara;

	surf1(img1,mascara, key_points1, descriptors1);
	surf2(img2, mascara, key_points2, descriptors2);

	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;
	matcher.match(descriptors1,descriptors2 ,matches);

	std::nth_element(matches.begin(),
		matches.begin()+29,
		matches.end());
	matches.erase(matches.begin()+30 ,matches.end());

	//显示
	namedWindow("SURF", WINDOW_AUTOSIZE);
	Mat img_matches;
	drawMatches(
		img1 ,key_points1,
		img2, key_points2,
		matches,
		img_matches,
		Scalar(255 ,255 ,255),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS
	);
	imshow("SURF", img_matches);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SURF_match.png", img_matches);

	return 0;
}