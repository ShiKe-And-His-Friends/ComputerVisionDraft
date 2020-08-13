#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
 * Mat content
 * **/
void MatContent () {
	Mat matArray[] = { Mat(1 ,2 ,CV_32FC1 ,cv::Scalar(1)) 
		,Mat(1 ,2 ,CV_32FC1 ,cv::Scalar(2))};
	Mat vout ,hout;
	vconcat(matArray ,2 ,vout);
	cout << "vertical is " << endl << vout << endl;
	hconcat(matArray ,2 ,hout);
	cout << "horizontal is " << endl << hout << endl;

	Mat A = (cv::Mat_<float>(2 ,2) << 1 ,7 ,2 ,8);
	Mat B = (cv::Mat_<float>(2 ,2) << 4 ,10 ,5 ,11);
	Mat vC ,hC;
	vconcat(A ,B ,vC);
	cout << "vertical is " << endl << vC << endl;
	hconcat(A ,B ,hC);
	cout << "horizontal is " << endl << hC << endl;

	Mat img00 = imread("/home/shike/Pictures/lenna_head_jpg_type.jpg");
	Mat img01 = imread("/home/shike/Pictures/lenna_head_jpg_type.jpg");
	Mat img10 = imread("/home/shike/Pictures/lenna_head_jpg_type.jpg");
	Mat img11 = imread("/home/shike/Pictures/lenna_head_jpg_type.jpg");
	Mat img ,img0 ,img1;
	hconcat(img00 ,img01 ,img0);
	hconcat(img10 ,img11 ,img1);
	vconcat(img1 ,img0 ,img);
	imshow("vertical horizontal" ,img);
	waitKey(0);
}

int main() {
	MatContent();	
	return 0;
}
