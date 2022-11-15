#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

void draw_matches(std::vector<KeyPoint> kps1, std::vector<KeyPoint> kps2 ,std::vector<DMatch> matches , Mat img1 ,Mat img2 ,Mat H ,Mat mask) {
	if (H.empty()){
		std::cout << "No homography found" << std::endl;
		return;
	}
	/***
		Mat matchesMask = mask.ravel().tolist()
		h, w, ch = img1.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]] ).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, H)
		#Ground truth transformation
		dst_GT = cv2.perspectiveTransform(pts, H_gt)
		img2_tr = cv2.polylines(decolorize(img2), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
		img2_tr = cv2.polylines(deepcopy(img2_tr), [np.int32(dst_GT)], True, (0, 255, 0), 3, cv2.LINE_AA)
		# Blue is estimated, green is ground truth homography
		draw_params = dict(matchColor = (255, 255, 0), # draw matches in yellow color
			singlePointColor = None,
			matchesMask = matchesMask, # draw only inliers
			flags = 2)
		img_out = cv2.drawMatches(decolorize(img1), kps1, img2_tr, kps2, tentatives, None, **draw_params)
		plt.figure(figsize = (12, 8))
		plt.imshow(img_out)
	***/
}

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

	/// ///////////// Ransac  ///////////////////////////////

	std::vector<Point2f> points1, points2;
	for (size_t i = 0; i < matches.size(); i++) {
		points1.push_back(keypoint1[matches[i].queryIdx].pt);
		points2.push_back(keypoint2[matches[i].trainIdx].pt);
	}
	
	Mat mask;
	Mat h = findHomography(points1 ,points2 ,mask ,RANSAC ,1.0);
	
	Mat prespectMat;
	warpPerspective(img1 ,prespectMat ,h ,img2.size());
	imshow("input" ,img2);
	imshow("algined Image", prespectMat);

	draw_matches(keypoint1, keypoint2, matches, img1, img2, h, mask);

	waitKey(6000);
	destroyAllWindows();


	/// ///////////// Py-Ransac  ///////////////////////////////


	return 0;
}