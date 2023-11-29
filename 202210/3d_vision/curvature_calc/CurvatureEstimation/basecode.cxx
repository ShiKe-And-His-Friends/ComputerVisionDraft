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

//����
int Dilate()
{
	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat image, image_gray, image_bw, image_dilate;   //��������ͼ�񣬻Ҷ�ͼ�񣬶�ֵͼ������ͼ��
	Mat rawImg = imread(image_url, -1);  //��ȡͼ��
	if (rawImg.empty())
	{
		cout << "��ȡ����" << endl;
		return -1;
	}
	// ���ͼ32λת�Ҷ�ͼ��
	image = convert32Fto8U(rawImg);

	//ת��Ϊ��ֵͼ
	image_gray = image;
	threshold(image_gray, image_bw, 120, 255, 0); //ͨ��0��1���ڶ�ֵͼ�񱳾���ɫ

	//��ʴ
	Mat se = getStructuringElement(0, Size(3, 3)); //������νṹԪ��
	dilate(image_bw, image_dilate, se, Point(-1, -1), 1); //ִ�����Ͳ���
	namedWindow("image_dilate", WINDOW_NORMAL);
	imshow("image_dilate", image_dilate);

	waitKey(0);  //��ͣ������ͼ����ʾ���ȴ���������
	cv::imwrite("..//Image//����.bmp", image_dilate);

	return 0;
}

//��ʴ
int Erosion()
{
	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat image, image_gray, image_bw, image_erosion;   //��������ͼ�񣬻Ҷ�ͼ�񣬶�ֵͼ�񣬸�ʴͼ��
	Mat rawImg = imread(image_url, -1);  //��ȡͼ��
	if (rawImg.empty())
	{
		cout << "��ȡ����" << endl;
		return -1;
	}
	// ���ͼ32λת�Ҷ�ͼ��
	image = convert32Fto8U(rawImg);

	//ת��Ϊ��ֵͼ
	image_gray = image;
	threshold(image_gray, image_bw, 100, 200, 0); //ͨ��0��1���ڶ�ֵͼ�񱳾���ɫ

	//��ʴ
	Mat se = getStructuringElement(1, Size(1, 1)); //������νṹԪ��
	erode(image_bw, image_erosion, se, Point(-1, -1), 1); //ִ�и�ʴ����

	namedWindow("image_erosion", WINDOW_NORMAL);
	imshow("image_erosion", image_erosion);
	waitKey(0);  //��ͣ������ͼ����ʾ���ȴ���������

	cv::imwrite("..//Image//��ʴ.bmp", image_erosion);

	return 0;
}

//����Բ���
int UseBaseHoughCricles(int argc, char** argv) {

	Mat src, mid_src, gray_src, dst;
	int min_threshold = 20;
	int max_range = 255;
	char gray_window[] = "gray_window";
	char Hough_result[] = "Hough_result";

	//Step1 ��ȡͼƬ
	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	// ���ͼ32λת�Ҷ�ͼ��
	src = convert32Fto8U(rawImg);

	if (src.empty()) {
		cout << "Could not load the image ...." << endl;
		return -1;
	}
	//imshow("input_image", src);

	//Step2 ���ڻ���Բ���������Ƚ����У�������Ҫʹ����ֵ�˲�/��˹�˲��ȷ���������ͼƬ
	//medianBlur(src, mid_src, 3);
	//imshow("��ֵ�˲�", mid_src);


	//Step3 ����ֵ�˲�֮���ͼƬת��Ϊ�Ҷ�ͼ
	//cvtColor(mid_src, gray_src, COLOR_BGR2GRAY);
	//imshow("�ҶȻ�", gray_src);

		// ��˹�˲�
	GaussianBlur(src, mid_src, Size(0, 0), 1.6);

	Canny(mid_src, gray_src, 200, 150);
	cv::imwrite("..//Image//HoughCricles2_Canny.bmp", gray_src);

	//Step4 ʹ�û���Բ���
	vector<Vec3f> pcircles;  //����һ��vector�����Բ����Ϣ��[0] [1]Բ�����꣬[2] �뾶��

	GaussianBlur(gray_src, gray_src, Size(3, 3), 2, 2);
	HoughCircles(gray_src, pcircles, HOUGH_GRADIENT, 1, gray_src.rows / 2, 60, 39); //, 800, 2200

	//Step5 ��Բ��ʾ��ԭͼ��
	gray_src.copyTo(dst);//��ԭͼ������dst
	for (size_t i = 0; i < pcircles.size(); i++) {
		Vec3f cc = pcircles[i];
		circle(dst, Point(cc[0], cc[1]), cc[2], Scalar(255), 2, LINE_AA);   //����Բ(ͼƬ����Բ��λ�ã��뾶����ɫ���߳�)
		circle(dst, Point(cc[0], cc[1]), 2, Scalar(255), 2, LINE_AA);       //����Բ��
	}

	namedWindow("Circle_image", WINDOW_NORMAL);
	imshow("Circle_image", dst);
	cv::imwrite("..//Image//HoughCricles2.bmp", dst);
	waitKey(0);

	std::cout << std::endl << std::endl << pcircles.size() << std::endl;
	return 0;
}

//Canny���Բ
int Canny_fit() {

	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	Mat src, dst, edge;
	// ���ͼ32λת�Ҷ�ͼ��
	src = convert32Fto8U(rawImg);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	// ��˹�˲�
	GaussianBlur(src, dst, Size(0, 0), 1.6);

	//Canny ���� ������ֵ60 ������ֵΪ25
	Canny(dst, edge, 25, 60);

	namedWindow("Canny", WINDOW_NORMAL);
	imshow("Canny", edge);
	waitKey(200);
	cv::imwrite("..//Image//Canny.bmp", edge);

	return 0;
}

//EDcricle���Բ
int EDcricle() {
	//int main() {

	std::cout << "Curvature..." << std::endl;

	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat rawImg = imread(image_url, -1);
	Mat testImg;

	// ���ͼ32λת�Ҷ�ͼ��
	testImg = convert32Fto8U(rawImg);

	cv::imwrite("..//Image//Img.bmp", testImg);

	EDCircles testEDCircles = EDCircles(testImg);
	Mat circleImg = testEDCircles.drawResult(false, ImageStyle::CIRCLES);

	EDPF testEDPF = EDPF(testImg);
	testEDCircles = EDCircles(testEDPF);

	vector<mCircle> circles = testEDCircles.getCircles();
	vector<mEllipse> ellipses = testEDCircles.getEllipses();

	std::cout << "Բ����:" << circles.size() << " ";
	std::cout << "��Բ����:" << ellipses.size() << " ";

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

//α-��С���˷����Բ
int LeastSquareMethod() {

	//��ȡ���ͼ ����Բ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	Mat src, dst, edge;
	// ���ͼ32λת�Ҷ�ͼ��
	src = convert32Fto8U(rawImg);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	// ��˹�˲�
	GaussianBlur(src, dst, Size(0, 0), 1.6);

	std::vector<std::vector<double>> points;

	//Canny ���� ������ֵ60 ������ֵΪ25
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

	//��С���˷�����Բ
	std::cout << "��ļ��� " << points.size() << std::endl;
	std::vector<double> center = leastSquaresCircle(points);

	std::cout << "���Բ��Բ��: (" << center[0] << ", " << center[1] << ")" << std::endl;

	return 0;
}


//Ԥ������EDcricle
int EDcricle_Pro() {

	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";

	Mat rawImg = imread(image_url, -1);
	Mat rawImgU8, testImg;

	// ���ͼ32λת�Ҷ�ͼ��
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

	std::cout << "Բ����:" << circles.size() << " ";
	std::cout << "��Բ����:" << ellipses.size() << " ";

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


//RANSAC���Բ
int Ransac_fit_cricle() {

	//��ȡ���ͼ ����Բ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	Mat src, dst, edge;
	// ���ͼ32λת�Ҷ�ͼ��
	src = convert32Fto8U(rawImg);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	// ��˹�˲�
	GaussianBlur(src, dst, Size(0, 0), 1.6);

	std::vector<std::vector<double>> points;

	//Canny ���� ������ֵ60 ������ֵΪ25
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

	//��С���˷�����Բ
	std::cout << "��ļ��� " << points.size() << std::endl;
	std::vector<double> center = leastSquaresCircle(points);

	std::cout << "���Բ��Բ��: (" << center[0] << ", " << center[1] << ")" << std::endl;

	return 0;
}

