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


int trackBarValue;
Mat img1 ,img2;

/**
 * track bar
 * **/

static void trackBarCallback (int ,void*) {
	float a = trackBarValue / 100.0;
	img2 = img1 * a;
	imshow("track bar light intensity" ,img2);
}

void TrackBar () {
	img1 = imread("/home/shike/Pictures/lenna_head_jpg_type.jpg");
	if (!img1.data) {
		cout << "image file failure." << endl;
		return;
	}
	namedWindow("track bar light intensity");
	imshow("track bar light intensity" ,img1);
	trackBarValue = 100;
	createTrackbar("xxx" ,"track bar light intensity" ,&trackBarValue ,600 ,trackBarCallback ,0);
	waitKey();
}

/**
 * CV color
 * **/
void CvColor() {
	Mat img = imread("/home/shike/Pictures/lenna_head_jpg_type.jpg");
	if (img.empty()) {
		cout << "picture date empty." << endl;
		return;
	}
	Mat gray ,HSV ,YUV ,Lab ,img32;
	img.convertTo(img32 ,CV_32F ,1.0/255);  // CV_8U convert to CV_32F

	cvtColor(img32 ,HSV ,COLOR_BGR2HSV);
	cvtColor(img32 ,YUV ,COLOR_BGR2YUV);
	cvtColor(img32 ,Lab ,COLOR_BGR2Lab);
	cvtColor(img32 ,gray ,COLOR_BGR2GRAY);
	imshow("raw" ,img32);
	imshow("HSV" ,HSV);
	imshow("YUV" ,YUV);
	imshow("lab" ,Lab);
	imshow("gray" ,gray);
	waitKey(0);
}

/**
 * DeepShallowCopy
 * **/
void DeepShallowCopy() {
	Mat img = imread("/home/shike/Pictures/lenna.jpg");
	Mat noobcv = imread("/home/shike/Pictures/me.png");
	if (img.empty() || noobcv.empty()) {
		cout << "picture data empty." << endl;
		return;
	}
	Mat ROI1 ,ROI2 ,ROI2_copy ,mask ,img2 , img_copy ,img_copy2;
	resize(noobcv ,mask , Size(200 ,200));
	img2 = img;  // light shallow
	
	img.copyTo(img_copy2);  // one of two deep shallow
	copyTo(img ,img_copy ,img);

	Rect rect(206 ,206 ,200 ,200);  // get ROI area in light shallow
	ROI1 = img(rect);
	ROI2 = img(Range(300 ,500) ,Range(300 ,500));

	img(Range(300 ,500) ,Range(300 ,500)).copyTo(ROI2_copy);  // get ROI area in deep shallow

	mask.copyTo(ROI1);  // add part picture
	
	imshow("added noobcv" ,img);
	imshow("ROI to ROI2" ,ROI2);
	imshow("deep shallow ROI2_copy" ,ROI2_copy);
	circle(img ,Point(300 ,300) ,20 ,Scalar(0 , 0 ,255) ,-1);
	imshow("light shallow" ,img2);
	imshow("deep shallow img_copy" ,img_copy);
	imshow("deep shallow img_copy2" ,img_copy2);
	imshow("Circle to ROI1 " ,ROI1);
	waitKey(0);
}

/**
 * Filp picture
 * **/
void FilpPicture () {
	Mat img = imread("/home/shike/Pictures/lenna_head_peng_type.png");
	if (img.empty()) {
		cout << "picture file empty." << endl;
		return;
	}
	Mat img_x ,img_y ,img_xy;
	flip(img ,img_x ,0);
	flip(img ,img_y ,1);
	flip(img ,img_xy ,-1);
	imshow("X Filp" ,img_x);
	imshow("Y Filp" ,img_y);
	imshow("XY Filp" ,img_xy);
	waitKey(0);
}

int main() {
	FilpPicture();
	return 0;
}

