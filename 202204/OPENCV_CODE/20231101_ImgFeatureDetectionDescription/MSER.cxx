#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src ,gray;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.png", -1);
	
	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	cvtColor(src ,gray ,COLOR_BGR2GRAY);
	
	//���������������ص㼯
	vector<vector<Point>> regions;
	vector<cv::Rect> mserBbox;

	//����MSER��
	Ptr<MSER> mserExtractor = MSER::create();
	mserExtractor->detectRegions(gray, regions ,mserBbox);

	//�ڻҶ�ͼ����������Բ�������
	for (int i = 0; i < regions.size();i++) {
		//ellipse(gray,fitEllipse(regions[i]) ,Scalar(255));
		drawContours(src, regions, i, Scalar(255,0,0), 4);
	}

	//��ʾ
	namedWindow("MSER", WINDOW_AUTOSIZE);
	imshow("MSER", src);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//MSER_2.png", src);

	return 0;
}