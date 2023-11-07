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

	// 把输入的彩色图像转换成灰度图像
	cvtColor(src ,gray ,cv::COLOR_BGR2GRAY);
	// 高斯滤波
	GaussianBlur(gray ,gray,Size(9,9) ,2 ,2);

	Mat corners, dilated_corners;
	//Kitchen-Rosefeld 角点检测，得到corners变量
	preCornerDetect(gray ,corners ,5);
	//使用3x3的结构元素进行数学形态学的膨胀处理，即在3x3的邻域找到最大值，结果存入
	//dilated_corners变量内
	dilate(corners ,dilated_corners ,Mat());

	//遍历图像的所有元素
	for (int j = 0; j< src.rows; j++) {
		// 每行的首地址指针
		const float* tmp_data = (const float*)dilated_corners.ptr(j);
		const float* corners_data = (const float*)corners.ptr(j);
		for (int i = 0; i < src.cols; i++) {
			// 非极大抑制，并且要满足阈值条件，阈值设0.037 ，膨胀处理的结果如果等于角点,
		// 说明该角点是在3x3领域内的最大值
			if (tmp_data[i] > 0.037 && corners_data[i]==tmp_data[i]) {
				// 在角点画一个椭圆
				circle(src ,Point(i ,j) ,5 ,Scalar(0 , 0,255) , -1 ,8 ,0);
			}
		}
		
		
	}

	namedWindow("Kitchen-Rosenfeld" ,WINDOW_AUTOSIZE);
	imshow("Kitchen-Rosenfeld",src);
	imwrite("Kitchen-Rosenfeld.png",src);

	return 0;
}