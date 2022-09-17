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

int main(int argc, char** argv) {

	std::cout << "test compile" << std::endl;

	// test opencv
	Mat m = imread("keycci2.png", IMREAD_UNCHANGED);
	imshow("mat", m);
	waitKey(1000);

	// test cuda
	GpuMat img1, img2;
	img1.upload(imread("qi.png", IMREAD_GRAYSCALE));
	img1.upload(imread("qi2.png", IMREAD_GRAYSCALE));

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