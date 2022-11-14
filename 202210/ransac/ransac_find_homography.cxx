#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

int main() {

	std::cout << "Ransac find homography." << std::endl;

	Mat img1 = imread("D:\\computerVisionAll\\ComputerVisionDraft\\202210\\ransac\\data\\adam1.png" ,COLOR_BGR2RGB);
	Mat img2 = imread("D:\\computerVisionAll\\ComputerVisionDraft\\202210\\ransac\\data\\adam2.png" , COLOR_BGR2RGB);
	imshow("1", img1);
	imshow("2", img2);
	waitKey(0);

	Ptr<SIFT> sift = SIFT::create(8000);

	std::vector<KeyPoint> keypoint1, keypoint2;
	Mat descriptor1, descriptor2;
	sift->detectAndCompute(img1 ,Mat() ,keypoint1 ,descriptor1);
	sift->detectAndCompute(img2, Mat(), keypoint2, descriptor2);

	std::vector<DMatch> mathes;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptor1, descriptor2, mathes, Mat());

	std::cout << "mathes " << mathes.size() << std::endl;

	return 0;
}