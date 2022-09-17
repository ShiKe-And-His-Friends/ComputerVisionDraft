#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

#include "opencv2/opencv_modules.hpp"  
#include "opencv2/core.hpp"  
#include "opencv2/features2d.hpp"  
#include "opencv2/highgui.hpp"  
#include "opencv2/cudafeatures2d.hpp"  
#include "opencv2/xfeatures2d/cuda.hpp"  


using namespace cv;
using namespace cv::cuda;

using namespace std;
int main()
{
    GpuMat img1, img2;
    img1.upload(imread("keycci.png", IMREAD_GRAYSCALE));
    img2.upload(imread("keycci2.png", IMREAD_GRAYSCALE));

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    SURF_CUDA surf;

    // detecting keypoints & computing descriptors  
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    // matching descriptors  
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    vector<DMatch> matches;
    matcher->match(descriptors1GPU, descriptors2GPU, matches);

    // downloading results  
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
    surf.downloadKeypoints(keypoints1GPU, keypoints1);
    surf.downloadKeypoints(keypoints2GPU, keypoints2);
    surf.downloadDescriptors(descriptors1GPU, descriptors1);
    surf.downloadDescriptors(descriptors2GPU, descriptors2);

    // drawing the results  
    Mat img_matches;
    drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);

    namedWindow("matches", 0);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}

int main2(int argc, char** argv) {

	std::cout << "test compile" << std::endl;

	// test opencv
	Mat m = imread("keycci.png", IMREAD_UNCHANGED);
	imshow("mat", m);
	waitKey(1000);

	// test cuda
	GpuMat img1, img2;
	img1.upload(imread("keycci.png", IMREAD_UNCHANGED));
	img1.upload(imread("keycci2.png", IMREAD_UNCHANGED));

	std::cout << "///////////////// cuda compile /////////////////" << std::endl;
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	SURF_CUDA surf;
	GpuMat keypoints1GPU, keypoints2GPU;
	GpuMat descriptors1GPU, descriptors2GPU;
	surf(img1 ,GpuMat() ,keypoints1GPU ,descriptors1GPU);
	surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);
	std::cout << "feature object1 :" << keypoints1GPU.cols << std::endl;
	std::cout << "feature object2 :" << keypoints2GPU.cols << std::endl;

	return 0;
}