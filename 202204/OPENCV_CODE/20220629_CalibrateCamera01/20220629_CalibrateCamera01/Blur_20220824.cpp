#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>


using namespace std;
using namespace cv;


int main(int argc ,char * argv[]) {

	cout  << "shikeDebug 20220824" << endl;
	Mat srcImg = imread(".//CameraData//PhotoData//Img-11.bmp" ,IMREAD_UNCHANGED);
	//imshow("srcImg" ,srcImg);


	// 高斯滤波处理图片
	Mat medianblurImg;
	medianBlur(srcImg ,medianblurImg ,9);
	imwrite(".//CameraData//blur20220824//srcImg.tiff", srcImg);
	imwrite(".//CameraData//blur20220824//medianblurImg.tiff" ,medianblurImg);

	Mat gaussianBlurImg;
	GaussianBlur(srcImg ,gaussianBlurImg ,Size(0,0) ,10 ,10);
	
	double dThresh = 0;
	double dAmount = 2;

	Mat maskImg = abs(srcImg - gaussianBlurImg) <= dThresh;
	Mat tempImg1 = srcImg * (1 + dAmount) + gaussianBlurImg * (-dAmount);

	imwrite(".//CameraData//blur20220824//gaussianblurImg.tiff" ,gaussianBlurImg);
	imwrite(".//CameraData//blur20220824//maskImg.tiff" ,maskImg);
	imwrite(".//CameraData//blur20220824//tempImg.tiff" ,tempImg1);

	Mat tempImg2;
	srcImg.copyTo(tempImg1,maskImg); // 辅图 掩码图
	srcImg.copyTo(tempImg2,maskImg); // 只掩码图
	imwrite(".//CameraData//blur20220824//mask03.tiff",tempImg1);
	imwrite(".//CameraData//blur20220824//mask04.tiff",tempImg2);


	// 低亮度增强
	Mat brightnessImg = srcImg.clone();
	int iLevel_L = 10, iLevel_R = 50;
	double dCoeff = 3.0;
	for (int i = 0; i < brightnessImg.rows; i++) {
		uchar* ptr = brightnessImg.ptr<uchar>(i);
		for (int j = 0; j < brightnessImg.cols;j++) {
			if (ptr[j] > iLevel_L && ptr[j] < iLevel_R) {
				ptr[j] = static_cast<uchar>((ptr[j] * dCoeff) < 255 ? (ptr[j] * dCoeff) : 255 );
			}
		}
	}
	imwrite(".//CameraData//blur20220824//brightnessImg.tiff", brightnessImg);


	// 圆环检测
	Mat blobCircleImg = srcImg.clone();
	SimpleBlobDetector::Params blobParams;
	
	blobParams.filterByArea = true;
	blobParams.minArea = 100;
	blobParams.maxArea = 7000;
	/*
	blobParams.filterByColor = false;
	blobParams.blobColor = 240;
	blobParams.minThreshold = 10;
	blobParams.maxThreshold = 200;
	blobParams.thresholdStep = 10;
	*/
	
	vector<KeyPoint> keyPoints;
	Ptr<FeatureDetector> blobDetector = SimpleBlobDetector::create(blobParams);
	blobDetector->detect(blobCircleImg ,keyPoints ,Mat()); 

	Mat blobCircleImg2;
	drawKeypoints(blobCircleImg ,keyPoints ,blobCircleImg2 ,Scalar(255 ,0 ,0) ,DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imwrite(".//CameraData//blur20220824//blobCircleImg.tiff" ,blobCircleImg2);


	//waitKey(0);

	return 0;
}
