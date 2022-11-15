#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

int main() {

	std::cout << "Ransac find homography." << std::endl;

	Mat img1 = imread("D:\\computerVisionAll\\ComputerVisionDraft\\202210\\ransac\\data\\adam1.png" ,COLOR_BGR2RGB);
	Mat img2 = imread("D:\\computerVisionAll\\ComputerVisionDraft\\202210\\ransac\\data\\adam2.png" , COLOR_BGR2RGB);
	
	cvtColor(img1 ,img1 ,COLOR_BGR2GRAY);
	cvtColor(img2, img2, COLOR_BGR2GRAY);

	Ptr<SIFT> sift = SIFT::create(800);

	std::vector<KeyPoint> keypoint1, keypoint2;
	Mat descriptor1, descriptor2;
	sift->detectAndCompute(img1 ,Mat() ,keypoint1 ,descriptor1);
	sift->detectAndCompute(img2, Mat(), keypoint2, descriptor2);

	std::cout << "detect1  " << descriptor1.size() << std::endl;
	std::cout << "detect2  " << descriptor2.size() << std::endl;
		
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::MatcherType::BRUTEFORCE_SL2);
	matcher->match(descriptor1, descriptor2, matches, Mat());
	std::sort(matches.begin() ,matches.end());
	std::cout << "mathes " << matches.size() << std::endl;
	
	Mat imMatches;
	drawMatches(img1 ,keypoint1 ,img2 ,keypoint2 ,matches ,imMatches);
	imshow("matches", imMatches);
	waitKey(1000);

	std::vector<Point2f> points1, points2;
	for (size_t i = 0; i < matches.size(); i++) {
		points1.push_back(keypoint1[matches[i].queryIdx].pt);
		points2.push_back(keypoint2[matches[i].trainIdx].pt);
	}
	
	Mat h = findHomography(points1 ,points2 ,RANSAC);
	Mat prespectMat;
	warpPerspective(img1 ,prespectMat ,h ,img2.size());
	imshow("input" ,img2);
	imshow("algined Image", prespectMat);
	waitKey(0);

	return 0;
}