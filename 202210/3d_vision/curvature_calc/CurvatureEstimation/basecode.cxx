#include <iostream>
#include <vector>
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

//膨胀
int Dilate()
{
	//读取深度图
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat image, image_gray, image_bw, image_dilate;   //定义输入图像，灰度图像，二值图像，膨胀图像
	Mat rawImg = imread(image_url, -1);  //读取图像；
	if (rawImg.empty())
	{
		cout << "读取错误" << endl;
		return -1;
	}
	// 深度图32位转灰度图像
	image = convert32Fto8U(rawImg);

	//转换为二值图
	image_gray = image;
	threshold(image_gray, image_bw, 120, 255, 0); //通过0，1调节二值图像背景颜色

	//腐蚀
	Mat se = getStructuringElement(0, Size(3, 3)); //构造矩形结构元素
	dilate(image_bw, image_dilate, se, Point(-1, -1), 1); //执行膨胀操作
	namedWindow("image_dilate", WINDOW_NORMAL);
	imshow("image_dilate", image_dilate);

	waitKey(0);  //暂停，保持图像显示，等待按键结束
	cv::imwrite("..//Image//膨胀.bmp", image_dilate);

	return 0;
}

//腐蚀
int Erosion()
{
	//读取深度图
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat image, image_gray, image_bw, image_erosion;   //定义输入图像，灰度图像，二值图像，腐蚀图像
	Mat rawImg = imread(image_url, -1);  //读取图像；
	if (rawImg.empty())
	{
		cout << "读取错误" << endl;
		return -1;
	}
	// 深度图32位转灰度图像
	image = convert32Fto8U(rawImg);

	//转换为二值图
	image_gray = image;
	threshold(image_gray, image_bw, 100, 200, 0); //通过0，1调节二值图像背景颜色

	//腐蚀
	Mat se = getStructuringElement(1, Size(1, 1)); //构造矩形结构元素
	erode(image_bw, image_erosion, se, Point(-1, -1), 1); //执行腐蚀操作

	namedWindow("image_erosion", WINDOW_NORMAL);
	imshow("image_erosion", image_erosion);
	waitKey(0);  //暂停，保持图像显示，等待按键结束

	cv::imwrite("..//Image//腐蚀.bmp", image_erosion);

	return 0;
}

//霍夫圆检测
int UseBaseHoughCricles(int argc, char** argv) {

	Mat src, mid_src, gray_src, dst;
	int min_threshold = 20;
	int max_range = 255;
	char gray_window[] = "gray_window";
	char Hough_result[] = "Hough_result";

	//Step1 读取图片
	//读取深度图
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	// 深度图32位转灰度图像
	src = convert32Fto8U(rawImg);

	if (src.empty()) {
		cout << "Could not load the image ...." << endl;
		return -1;
	}
	//imshow("input_image", src);

	//Step2 由于霍夫圆检测对噪声比较敏感，所以需要使用中值滤波/高斯滤波等方法，处理图片
	//medianBlur(src, mid_src, 3);
	//imshow("中值滤波", mid_src);


	//Step3 将中值滤波之后的图片转化为灰度图
	//cvtColor(mid_src, gray_src, COLOR_BGR2GRAY);
	//imshow("灰度化", gray_src);

		// 高斯滤波
	GaussianBlur(src, mid_src, Size(0, 0), 1.6);

	Canny(mid_src, gray_src, 200, 150);
	cv::imwrite("..//Image//HoughCricles2_Canny.bmp", gray_src);

	//Step4 使用霍夫圆检测
	vector<Vec3f> pcircles;  //创建一个vector，存放圆的信息（[0] [1]圆心坐标，[2] 半径）

	GaussianBlur(gray_src, gray_src, Size(3, 3), 2, 2);
	HoughCircles(gray_src, pcircles, HOUGH_GRADIENT, 1, gray_src.rows / 2, 60, 39); //, 800, 2200

	//Step5 将圆显示在原图上
	gray_src.copyTo(dst);//将原图拷贝给dst
	for (size_t i = 0; i < pcircles.size(); i++) {
		Vec3f cc = pcircles[i];
		circle(dst, Point(cc[0], cc[1]), cc[2], Scalar(255), 2, LINE_AA);   //绘制圆(图片名，圆心位置，半径，颜色，边长)
		circle(dst, Point(cc[0], cc[1]), 2, Scalar(255), 2, LINE_AA);       //绘制圆心
	}

	namedWindow("Circle_image", WINDOW_NORMAL);
	imshow("Circle_image", dst);
	cv::imwrite("..//Image//HoughCricles2.bmp", dst);
	waitKey(0);

	std::cout << std::endl << std::endl << pcircles.size() << std::endl;
	return 0;
}

//Canny拟合圆
int Canny_fit() {

	//读取深度图
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

	//Canny 方法 ，高阈值60 ，低阈值为25
	Canny(dst, edge, 25, 60);

	namedWindow("Canny", WINDOW_NORMAL);
	imshow("Canny", edge);
	waitKey(200);
	cv::imwrite("..//Image//Canny.bmp", edge);

	return 0;
}

//EDcricle拟合圆
int EDcricle() {
	//int main() {

	std::cout << "Curvature..." << std::endl;

	//读取深度图
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat rawImg = imread(image_url, -1);
	Mat testImg;

	// 深度图32位转灰度图像
	testImg = convert32Fto8U(rawImg);

	cv::imwrite("..//Image//Img.bmp", testImg);

	EDCircles testEDCircles = EDCircles(testImg);
	Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);

	EDPF testEDPF = EDPF(testImg);
	testEDCircles = EDCircles(testEDPF);

	vector<mCircle> circles = testEDCircles.getCircles();
	vector<mEllipse> ellipses = testEDCircles.getEllipses();

	std::cout << "圆数量:" << circles.size() << " ";
	std::cout << "椭圆数量:" << ellipses.size() << " ";

	circleImg = testEDCircles.drawResult(true, ImageStyle::BOTH);
	namedWindow("CIRCLES and ELLIPSES RESULT IMAGE", WINDOW_NORMAL);
	imshow("CIRCLES and ELLIPSES RESULT IMAGE", circleImg);
	waitKey();

	int noCircles = testEDCircles.getCirclesNo();
	std::cout << "Number of circles: " << noCircles << " " << std::endl;
	if (noCircles >= 1) {
		Point2d nCenter = circles[0].center;
		double radies = circles[0].r;
		std::cout << "X " << nCenter.x << " Y" << nCenter.y << " Raides " << radies << std::endl;
	}

	std::cout << "Done..." << std::endl;

	return 0;
}

//伪-最小二乘法拟合圆
int LeastSquareMethod() {

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
	std::vector<double> center = leastSquaresCircle(points);

	std::cout << "拟合圆的圆心: (" << center[0] << ", " << center[1] << ")" << std::endl;

	return 0;
}


//预处理后的EDcricle
int EDcricle_Pro() {

	//读取深度图
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat rawImg = imread(image_url, -1);
	Mat rawImgU8, testImg;

	// 深度图32位转灰度图像
	rawImgU8 = convert32Fto8U(rawImg);

	Mat gaussian_mat, canny_mat;
	GaussianBlur(rawImgU8, gaussian_mat, Size(0, 0), 1.6);
	Canny(gaussian_mat, canny_mat, 200, 150);
	GaussianBlur(canny_mat, testImg, Size(7, 7), 2, 2);

	EDCircles testEDCircles = EDCircles(testImg);
	Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);

	EDPF testEDPF = EDPF(testImg);
	testEDCircles = EDCircles(testEDPF);

	vector<mCircle> circles = testEDCircles.getCircles();
	vector<mEllipse> ellipses = testEDCircles.getEllipses();

	std::cout << "圆数量:" << circles.size() << " ";
	std::cout << "椭圆数量:" << ellipses.size() << " ";

	circleImg = testEDCircles.drawResult(true, ImageStyle::BOTH);
	namedWindow("CIRCLES and ELLIPSES RESULT IMAGE", WINDOW_NORMAL);
	imshow("CIRCLES and ELLIPSES RESULT IMAGE", circleImg);
	waitKey();

	int noCircles = testEDCircles.getCirclesNo();
	std::cout << "Number of circles: " << noCircles << " " << std::endl;
	if (noCircles >= 1) {
		Point2d nCenter = circles[0].center;
		double radies = circles[0].r;
		std::cout << "X " << nCenter.x << " Y" << nCenter.y << " Raides " << radies << std::endl;
	}

	return 0;
}


//RANSAC拟合圆
int Ransac_fit_cricle() {

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
	std::vector<double> center = leastSquaresCircle(points);

	std::cout << "拟合圆的圆心: (" << center[0] << ", " << center[1] << ")" << std::endl;

	return 0;
}

