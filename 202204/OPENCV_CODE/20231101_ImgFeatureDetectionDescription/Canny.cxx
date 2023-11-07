#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

int main(int argc, char** argv) {

	Mat src, dst, edge;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	// ��˹�˲�
	GaussianBlur(src, dst, Size(0,0),1.6);

	//Canny ���� ������ֵ60 ������ֵΪ25
	Canny(dst ,edge ,25 ,60);

	namedWindow("Canny", WINDOW_AUTOSIZE);
	imshow("Canny", edge);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//Canny.png", edge);

	return 0;
}