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

	SIFT sift; //ʵ����SIFT��

	vector<KeyPoint> key_points; //������
	// descriptors Ϊ������ mascaraΪ�������
	Mat descriptors, mascara;
	Mat output_img; //���ͼ�����

	sift(src ,mascara ,key_points , descriptors); // ִ��SIFT����

	// �����ͼ���л���������
	drawKeypoints(
		src,
		key_points, //������ʸ��
		output_img, //���ͼ��
		Scalar::all(-1), //�������������ɫ��Ϊ���
		//��������Ϊ���Ļ�Բ��Բ�İ뾶��ʾ������Ĵ�С��ֱ�߱�ʾ������ķ���
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//��ʾ
	namedWindow("SIFT", WINDOW_AUTOSIZE);
	imshow("SIFT", output_img);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SIFT.png", output_img);

	return 0;
}