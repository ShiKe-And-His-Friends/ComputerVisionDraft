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

	// ��ƥ�������ͼ������img1����img2��Ҳ������img1��ʶ��img2
	Mat img1 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.jpg", -1);
	Mat img2 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.jpg", -1);

	if (!img1.data || !img2.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	SIFT sift1 ,sift2; //ʵ����SIFT��

	vector<KeyPoint> key_points1 ,key_points2; //������
	// descriptors Ϊ������ mascaraΪ�������
	Mat descriptors1, descriptors2, mascara;

	sift(img1, mascara, key_points1, descriptors1); // ִ��SIFT����
	sift(img2, mascara, key_points2, descriptors2);

	//ʵ��������ƥ����-BruteForceMatcher
	BruteForceMatcher<L2 <float>> matcher;
	//����ƥ��������
	vector<DMatch> matches;
	//ʵ��������֮���ƥ�䣬�õ�����matches
	matcher.match(descriptors1 ,descriptors2 ,matches);

	// ��ȡǰ30�����ƥ����
	std::nth_element(matches.begin(),
		matches.begin()+29,
		matches.end());
	//�޳��������ƥ����
	matches.erase(matches.begin()+30 ,matches.end());

	//��ʾ
	namedWindow("SIFT_matches", WINDOW_AUTOSIZE);
	Mat img_matches;
	// �����ͼ���л���������
	drawMatches(
		img1,key_points1, //��һ��ͼ��������ʸ��
		img2, key_points2, //�ڶ���ͼ��������ʸ��
		matches, //ƥ��������
		img_matches,//ƥ�����ͼ��
		Scalar(255 ,255,255), //�ð�ɫֱ����������ͼ���е�������
		//��������Ϊ���Ļ�Բ��Բ�İ뾶��ʾ������Ĵ�С��ֱ�߱�ʾ������ķ���
		);

	imshow("SIFT_matches", img_matches);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SIFT_matches.png", img_matches);

	return 0;
}