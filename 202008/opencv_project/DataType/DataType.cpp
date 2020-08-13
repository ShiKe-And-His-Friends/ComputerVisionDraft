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

/**
 * LUT look up table
 * **/

void LookUpTable () {
	uchar lutFirst[256];
	for (int i = 0 ; i < 256 ; i ++) {
		if (i <= 100) {
			lutFirst[i] = 0;
		}
		if (i > 100 && i <= 200) {
			lutFirst[i] = 100;
		}
		if (i > 200) {
			lutFirst[i] = 255;
		}
	}
	Mat lutOne(1 ,256 ,CV_8UC1 ,lutFirst);
	
	uchar lutSecond[256];
	for (int i = 0 ; i < 256 ; i ++) {
		if (i <= 100) {
			lutSecond[i] = 0;
		}
		if (i > 100 && i <= 150) {
			lutSecond[i] = 100;
		}
		if (i > 150 && i <= 200) {
			lutSecond[i] = 150;
		}
		if (i > 200) { 
			lutSecond[i] = 255;
		}
	}
	Mat lutTwo(1 ,256 ,CV_8UC1 ,lutSecond);

	uchar lutThird[256];
	for (int i = 0 ; i < 256 ; i++) {
		if (i <= 100) {
			lutThird[i] = 100;
		}
		if (i > 100 && i <= 200) {
			lutThird[i] = 200;
		}
		if (i > 200) {
			lutThird[i] = 255;
		}
	}
	Mat lutThree(1 ,256 ,CV_8UC1 ,lutThird);

	vector<Mat> mergeMats;
	mergeMats.push_back(lutOne);
	mergeMats.push_back(lutTwo);
	mergeMats.push_back(lutThree);
	Mat LutTree;
	merge(mergeMats ,LutTree);
	Mat img = imread("/home/shike/Pictures/lenna_head_peng_type.png");
	if (img.empty()) {
		cout << "picture file empty." << endl;
	}
	Mat gray ,out0 ,out1 ,out2;
	cvtColor(img ,gray ,COLOR_BGR2GRAY);
	LUT(gray ,lutOne ,out0);
	LUT(img ,lutTwo ,out1);
	LUT(img ,lutThree ,out2);
	imshow("out0" ,out0);
	imshow("out1" ,out1);
	imshow("out2" ,out2);
	waitKey(0);
}

/**
 * LogicOperation
 * **/
void LogicOperation () {
	Mat img = imread("/home/shike/Pictures/lenna_head_peng_type.png");
	if (img.empty()) {
                 cout << "picture file empty." << endl;
         }
	Mat img0 = Mat::zeros(200 ,200 ,CV_8UC1);
	Mat img1 = Mat::zeros(200 ,200 ,CV_8UC1);
	Rect rect0(50 ,50 ,100 ,100);
	img0(rect0) = Scalar(255);
	Rect rect1(100 ,100 ,100 ,100);
	img1(rect1) = Scalar(255);
	imshow("img0" ,img0);
	imshow("img1" ,img1);

	Mat myAnd ,myOr ,myXor ,myNot ,imgNot;
	bitwise_not(img0 ,myNot);
	bitwise_and(img0 ,img1 ,myAnd);
	bitwise_or(img0 ,img1 ,myOr);
	bitwise_xor(img0 ,img1 ,myXor);
	bitwise_not(img ,imgNot);
	imshow("myAnd" ,myAnd);
	imshow("myOr" ,myOr);
	imshow("myXor" ,myXor);
	imshow("myNot" ,myNot);
	imshow("imgNot" ,imgNot);
	waitKey(0);
}

int main() {
	LogicOperation();
	return 0;
}

