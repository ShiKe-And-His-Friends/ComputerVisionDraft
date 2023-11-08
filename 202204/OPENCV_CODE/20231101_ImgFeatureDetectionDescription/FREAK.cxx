
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

	Ptr<BRISK> brisk = cv::Feature2D::create(60, 4, 1.0); // 实例化BRISK
	FREAK freak; //实例化FREAK

	vector<KeyPoint> key_points1 , key_points2;
	
	brisk->detect(img1, key_points1);
	brisk->detect(img2, key_points2);//BRISK的特征点描述方法

	Mat descriptors1, descriptors2;
	freak.compute(img1 ,key_points1 ,descriptors1); //创建FREAK描述符
	freak.compute(img2, key_points2, descriptors2);

	BFMatcher<Hanmming> matcher;
	vector<DMatch> matches;
	matcher.match(descriptors1 ,descriptors2 ,matches);

	std::nth_element(
		matches.begin(),
		matches.begin()+29,
		matches.end());
	matches.erase(matches.begin()+30 ,matches.end());

	//显示
	namedWindow("FREAK_match", WINDOW_AUTOSIZE);
	Mat output_img;
	drawMatches(
		img1,
		key_points1,
		img2,
		key_points2,
		matches,
		output_img,
		Scalar(255,255,255));
	imshow("FREAK_match", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//FREAK_match.png", output_img);

	return 0;
}