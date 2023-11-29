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

//����Բ���
int CurvatureEstimation() {
//int main(){

	Mat src, mid_src, gray_src, dst;
	//��ȡ���ͼ
	string image_url = "E:/project/EDCircle/pic_20231123094657551_tif32.tif";
	Mat rawImg = imread(image_url, -1);

	// ���ͼ32λת�Ҷ�ͼ��
	src = convert32Fto8U(rawImg);

	if (src.empty()) {
		cout << "Could not load the image ...." << endl;
		return -1;
	}

	// ��˹�˲�
	GaussianBlur(src, mid_src, Size(0, 0), 1.6);
	Canny(mid_src, gray_src, 200, 150);

	//ʹ�û���Բ���
	vector<Vec3f> pcircles;  //����һ��vector�����Բ����Ϣ��[0] [1]Բ�����꣬[2] �뾶��
	GaussianBlur(gray_src, gray_src, Size(3, 3), 2, 2);
	HoughCircles(gray_src, pcircles, HOUGH_GRADIENT,1, gray_src.rows / 2, 60, 39); //, 800, 2200
	
	//Step5 ��Բ��ʾ��ԭͼ��
	cvtColor(gray_src , dst, COLOR_GRAY2RGB);
	std::cout << std::endl << std::endl << "��⵽Բ������ " << pcircles.size() << std::endl;
	for (size_t i = 0; i < pcircles.size(); i++) {
		Vec3f cc = pcircles[i];
		circle(dst, Point(cc[0], cc[1]), cc[2], Scalar(0,0,255), 2, LINE_AA);   //����Բ(ͼƬ����Բ��λ�ã��뾶����ɫ���߳�)
		circle(dst, Point(cc[0], cc[1]), 4, Scalar(0,255,0), 2, LINE_AA);       //����Բ��
	}


	if (pcircles.size() >=1) {
		Vec3f data = pcircles[0];
		float x = data[0];
		float y = data[1];
		float R = data[2];
		std::cout << "Բ������ " << x << " " << y << " �뾶����" << R << std::endl;
	}

	//����Բ����������
	Mat srctorImage;
	src.copyTo(srctorImage);
	cvtColor(srctorImage , srctorImage ,COLOR_GRAY2RGB);

	//���ýǶ�
	float INTIAL_ANGLE = RADIUS_ANGLE;
	vector<Point3f> lines_coordinate;
	if (pcircles.size() <1) {
		std::cout << "δ��⵽Բ." << std::endl;
		return -1;
	}

	//���ĵ�
	Vec3f center_point(2316, 1950, 1570);
	//Vec3f center_point = pcircles[0];
	circle(srctorImage, Point(floor(center_point[0]), floor(center_point[1])), center_point[2], Scalar(255,0,0), 5, LINE_AA);

	//�жϽǶ��Ƿ����ص�
	bool is_overlap = 3600 % static_cast<unsigned int>(INTIAL_ANGLE*10) == 0;

	for (int i = 0; i * INTIAL_ANGLE < (is_overlap ? 180 : 360); i++ ) {
		float angle = i * INTIAL_ANGLE;
		
		std::cout << "�Ƕ�" << angle << std::endl;

		// ��ջ�������
		lines_coordinate.clear();
		
		//��ֵ����ֱ���ϵ�����ֵ
		LineEstimation(rawImg,center_point, lines_coordinate, angle);
	
		//��¼������Ϣ
		ofstream out_file("..//Image//��Բ���//" +to_string(angle) + ".a" ,ios::out | ios::binary | ios::trunc);

		//��������
		Scalar color(rand() % 255, rand() % 255, rand() % 255);
		for (int k = 0; k < lines_coordinate.size(); k++) {
			Point3f point = lines_coordinate[k];
			circle(srctorImage, Point(floor(point.x), floor(point.y)), 1, color, 2, LINE_AA);
			out_file << point.z << std::endl;
		}
		out_file.close();

		//���㺣�׹�ʽ����
		double curvity = getArcCurvity(lines_coordinate);
		std::cout <<"�������� " << curvity << std::endl;
	}
	
	namedWindow("Circle_image", WINDOW_NORMAL);
	imshow("Circle_image", srctorImage);
	waitKey(1000);
	cv::imwrite("..//Image//temp.bmp", srctorImage);

	return 0;
}