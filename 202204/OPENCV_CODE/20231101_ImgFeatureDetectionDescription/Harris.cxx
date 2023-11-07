#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

int main(int argc, char** argv) {

	Mat src, src_gray;
	Mat dst, dst_norm, dst_norm_scaled;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	cvtColor(src ,src_gray ,COLOR_BGR2GRAY);

	int blockSize = 2; //均值窗口的大小
	int apertureSize = 3; //Sobel的孔径尺寸
	double k = 0.04; //经验系数
	int thresh = 155; //里面的阈值
	// Harris角点检测，输出图像中每个像素的R值
	cornerHarris(src_gray ,dst ,blockSize ,apertureSize ,k ,BORDER_DEFAULT);

	//归一化R图像，使其在0到255
	normalize(dst ,dst_norm ,0 ,255 ,NORM_MINMAX ,CV_32FC1 ,Mat());
	//把R图像转换成CV_8U，以便能显示出来
	convertScaleAbs(dst_norm ,dst_norm_scaled);

	//在角点处画一个圆圈
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols;i ++) {
			if ((int)dst_norm.at<float>(j ,i) > thresh) {
				circle(src ,Point(i ,j) ,5 ,Scalar(0 ,0 ,255) ,1 ,8 ,0);
			}
		}
	}

	namedWindow("Harris", WINDOW_AUTOSIZE);
	imshow("Harris", src);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//Harris.png", src);

	return 0;
}