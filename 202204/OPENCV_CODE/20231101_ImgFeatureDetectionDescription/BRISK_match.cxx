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

	//FAST的阈值设置60 ，金字塔采用4个层组，采样模板的基准尺寸是1.0
	Ptr<Feature2D> brisk_detect1 = BRISK::create(60, 4, 1.0);
	Ptr<Feature2D> brisk_detect2 = BRISK::create(60, 4, 1.0);
	vector<KeyPoint> key_points1, key_points2;

	Mat descriptors1, descriptors2, mascara;
	Mat output_img;

	brisk_detect1->detectAndCompute(img1 , mascara,key_points1 ,descriptors1);
	brisk_detect2->detectAndCompute(img2, mascara, key_points2, descriptors2);

	//采用汉明距离匹配方法
	//Hamming matcher(Hamming);
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING,false);
	vector<DMatch> matches;

	matcher->match(descriptors1 , descriptors2 , matches);
	
	std::nth_element(
		matches.begin(),
		matches.begin()+29,
		matches.end());
	matches.erase(matches.begin()+30 ,matches.end());

	//显示
	namedWindow("BRISK_match", WINDOW_AUTOSIZE);
	drawMatches(
		img1,
		key_points1,
		img2,
		key_points2,
		matches,
		output_img,
		Scalar(255,255,255));
	imshow("BRISK_match", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//BRISK_match.png", output_img);

	return 0;
}