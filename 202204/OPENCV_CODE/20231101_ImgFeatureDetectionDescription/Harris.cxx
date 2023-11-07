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

	int blockSize = 2; //��ֵ���ڵĴ�С
	int apertureSize = 3; //Sobel�Ŀ׾��ߴ�
	double k = 0.04; //����ϵ��
	int thresh = 155; //�������ֵ
	// Harris�ǵ��⣬���ͼ����ÿ�����ص�Rֵ
	cornerHarris(src_gray ,dst ,blockSize ,apertureSize ,k ,BORDER_DEFAULT);

	//��һ��Rͼ��ʹ����0��255
	normalize(dst ,dst_norm ,0 ,255 ,NORM_MINMAX ,CV_32FC1 ,Mat());
	//��Rͼ��ת����CV_8U���Ա�����ʾ����
	convertScaleAbs(dst_norm ,dst_norm_scaled);

	//�ڽǵ㴦��һ��ԲȦ
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