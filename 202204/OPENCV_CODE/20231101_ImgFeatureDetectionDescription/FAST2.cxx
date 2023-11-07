#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	Mat src, gray;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg", -1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	//��ɫת��ɫ
	cvtColor(src ,gray ,COLOR_BGR2GRAY);

	//����������KeyPoint����
	vector<KeyPoint> keyPoints;

	//����FAST��������ֵ����55
	FAST(gray ,keyPoints ,55);

	int total = keyPoints.size();
	for (int i = 0; i < total; i++) {
		//response������FAST�㷨��ָ���ǽǵ���Ӧֵ���ǽǵ����Ӧֵ��0
		if (keyPoints[i].response != 0) {
			circle(src ,Point((int) keyPoints[i].pt.x ,(int)keyPoints[i].pt.y) ,5 ,Scalar(0,0,255),-1,8,0);
		}
	}

	//��ʾ
	namedWindow("FAST", WINDOW_AUTOSIZE);
	imshow("FAST", src);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//FAST.png", src);

	return 0;
}