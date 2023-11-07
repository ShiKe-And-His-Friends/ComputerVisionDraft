#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/Xfeatures2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src, src_gray;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	SIFT sift; //实例化SIFT类

	vector<KeyPoint> key_points; //特征点
	// descriptors 为描述符 mascara为掩码矩阵
	Mat descriptors, mascara;
	Mat output_img; //输出图像矩阵

	sift(src ,mascara ,key_points , descriptors); // 执行SIFT运算

	// 在输出图像中绘制特征点
	drawKeypoints(
		src,
		key_points, //特征点矢量
		output_img, //输出图像
		Scalar::all(-1), //绘制特征点的颜色，为随机
		//以特征点为中心画圆，圆的半径表示特征点的大小，直线表示特征点的方向
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//显示
	namedWindow("SIFT", WINDOW_AUTOSIZE);
	imshow("SIFT", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SIFT.png", output_img);

	return 0;
}