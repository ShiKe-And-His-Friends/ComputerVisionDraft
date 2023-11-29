#include <iostream>
#include <vector>
#include <fstream>
#include "ImageDetect.hpp"
#include "CurvatureEstimation.hpp"
#include "EDLib.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int main() {
	
	std::cout << "Range image to cloud points..." << std::endl;


	//读取深度图 输入圆
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	Mat src, dst, edge;
	// 深度图32位转灰度图像
	src = convert32Fto8U(rawImg);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	// 高斯滤波
	GaussianBlur(src, dst, Size(0, 0), 1.6);

	std::vector<std::vector<double>> points;

	//Canny 方法 ，高阈值60 ，低阈值为25
	Canny(dst, edge, 25, 60);

	for (int j = 0; j < edge.rows; j++) {
		for (int i = 0; i < edge.cols; i++) {

			unsigned char values = edge.at<unsigned char>(j, i);
			if (values == 255) {
				std::vector<double> point;
				point.push_back(i);
				point.push_back(j);
				points.push_back(point);
			}

		}
	}

	//最小二乘法计算圆
	std::cout << "点的集合 " << points.size() << std::endl;

	return 0;
}