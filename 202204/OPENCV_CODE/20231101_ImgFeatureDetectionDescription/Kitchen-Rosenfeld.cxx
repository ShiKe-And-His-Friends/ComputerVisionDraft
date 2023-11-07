#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

int main(int argc ,char** argv) {

	Mat src, gray, color_edge;
	src = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//building.jpg",-1);

	if (!src.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	// ������Ĳ�ɫͼ��ת���ɻҶ�ͼ��
	cvtColor(src ,gray ,cv::COLOR_BGR2GRAY);
	// ��˹�˲�
	GaussianBlur(gray ,gray,Size(9,9) ,2 ,2);

	Mat corners, dilated_corners;
	//Kitchen-Rosefeld �ǵ��⣬�õ�corners����
	preCornerDetect(gray ,corners ,5);
	//ʹ��3x3�ĽṹԪ�ؽ�����ѧ��̬ѧ�����ʹ�������3x3�������ҵ����ֵ���������
	//dilated_corners������
	dilate(corners ,dilated_corners ,Mat());

	//����ͼ�������Ԫ��
	for (int j = 0; j< src.rows; j++) {
		// ÿ�е��׵�ַָ��
		const float* tmp_data = (const float*)dilated_corners.ptr(j);
		const float* corners_data = (const float*)corners.ptr(j);
		for (int i = 0; i < src.cols; i++) {
			// �Ǽ������ƣ�����Ҫ������ֵ��������ֵ��0.037 �����ʹ���Ľ��������ڽǵ�,
		// ˵���ýǵ�����3x3�����ڵ����ֵ
			if (tmp_data[i] > 0.037 && corners_data[i]==tmp_data[i]) {
				// �ڽǵ㻭һ����Բ
				circle(src ,Point(i ,j) ,5 ,Scalar(0 , 0,255) , -1 ,8 ,0);
			}
		}
		
		
	}

	namedWindow("Kitchen-Rosenfeld" ,WINDOW_AUTOSIZE);
	imshow("Kitchen-Rosenfeld",src);
	imwrite("Kitchen-Rosenfeld.png",src);

	return 0;
}