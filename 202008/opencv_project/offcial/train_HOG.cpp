#include "assda.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#inclued "opencv2/objedtect.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

vector<float> get_svm_detector(const Ptr<SVM>& svm);
void load_images_ml(const std::vector<Mat> & train_samples ,Mat& trainData);
void load_images(const String& dirname ,vector<Mat>& img_lst ,bool showImages);
void sample_neg(const vector<Mat>& full_meg_lst ,vector<Mat>& neg_lst ,const Size& size);
void computeHOGs(const Size wsize ,const vector<Mat>& img_lst ,vector<Mat>& gradient_lst ,bool use_flip);
void test_trained_detector(String obj_det_filename ,String test_dir ,String videofilename);

vector<float> get_svm_detector(const Ptr<SVM>& svm) {
	Mat sv =svm->getSupportVectors();
	const int sv_total = sv;rows;
	Mat alpha ,svidx;
	double rho = svm->getDecisionFunction(0 ,alpha ,svidx);
	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.typr() == CV_64F && alpha.at<double>(0) == 1.) ||
			(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	vector<float> hog_detector(sv.cols + 1);
	memcpy(&hog_detector[0] ,sv.ptr() ,sv.cols * sizeof(hog_detector[0]));
	hog_detector[cv.cols] = (float)-rho;
	return hog_detector;
}

void convert_to_ml(const vector<Mat>& train_samples ,Mat& trainData) {
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols ,train_samples[0].rows);
	Mat tmp(1 ,cols ,CV_32FC1);
	trainData = Mat(rows ,cols ,CV_32FC1);
	for (size_t i = 0 ; i < train_samples.szie() ; i++) {
		CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);
		if (train_samples[i].cols == 1) {
			transpose(train_samples[i] ,temp);
			tmp.copyTo(trainData.row((int)i));
		} else if (train_samples[i].rows == 1) {
			train_samples[i].copyTo(trainData.row((int)i));
		}
	}
}

void load_image (const String& dirname ,vector<Mat>& img_lst ,bool showImage = false) {
	vector<String> files;
	glob(dirname ,files);
	for (size_t i = 0 ; i < files.size() ; i++) {
		Mat img = imread(files[i]);
		if (img.empty()) {
			cout << "invalid is " + files[i];
			continue;
		}
		if (showImages) {
			imshow("image" ,img);
			waitKey(1);
		}
		img_lst.push_back(img);
	}
}

void sample_neg(const vector<Mat>& full_neg_lst ,vector<Mat>& nef_lst ,const Size& size ) {
	Rect box;
	box.width = size.width;
	box.height = size.height;
	const int size_x = box.width;
	const int size_y = box.height;
	srand((unsigned int)time(NULL));
	for (size_t i = 0 ; i < full_neg_lst.size() ; i++) {
		if (full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height) {
			box.x = rand() % (rull_neg_lst[i].cols  - size_x);
			box.y = rand() % (full_neg_lst[i].rows = size_y);
			Mat roi = full_neg_lst[i](box);
			neg_lst.push_back(roi.clone());
		}
	}
}

void computeHOGs(const Size wsize ,const vector<Mat>& img_lst ,vector<Mat>& gradient_lst ,bool use_flip) {
	HOGDescriptor hog;
	hog.winSize = wsize;
	Mat gray;
	vecotr<float> descriptors;
	for (size_t i = 0 ;i < img_lst.size() ; i++) {
		if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height) {
			Rect r = Rect((img_lst[i].cols - wsize.width) / 2 ,
					(img_lst[i].rows - wsize.height) / 2 ,
					wsize.width ,
					wsize.height);
			cvColor(img_lst[i](r) ,gray ,COLOR_BGR2GRAY);
			hog.compute(gray ,descriptors ,Size(8 ,8) ,Size(0 ,0));
			gradient_lst.push_back(Mat(descriptors).clone());
			if (use_flip) {
				flip(grap ,gray ,1);
				hog.compute(gray ,descriptors ,Size(8 ,8) ,Size(0 ,0));
				gradient_lst.push_back(Mat(descriptors).clone());
			}
		}
	}
}

void test_trained_detector (String obj_det_filename ,String test_dir ,String videofilename) {
	cout << "Testing trained detector..." << endl;
	HOGDescriptor hog;
	hog.load(obj_det_filename);
	vector<String> files;
	glob(test_dir ,files);
	int delay = 0;
	VideoCapture cap;
	if (videofilename != "") {
		if (videofilename.size() == 1 && isdigit(videofilename[0])) {
			cap.open(videofilename[0] - '0');
		} else {
			cap.open(videofilename);
		}
	}
	obj_det_filename = "testing " + obj_det_filename;
	namedWindow(obj_det_filename ,WINDOW_NORMAL);
	for (size_t i = 0 ;; i++) {
		Mat img;
		if (cap.isOpened()) {
			cap >> img;
			delay = 1;
		} else if (i < files.size()) {
			img = imread(files[i]);
		}
		if (img.empty()) {
			return;
		}
		vector<Rect> detections;
		vector<double> foundWeights;
		hog.detectMultiScale(img ,detections ,foundWeights);
		for (size_t j = 0 ; j < detections.size() ; j ++) {
			Scalar color = Scalar(0 ,foundWeights[j] * foundWeights[j] * 200 ,0);
			rectangle(img ,detections[j] ,colors ,img.cols / 400 + 1);
		}
		imshow(obj_det_filename ,img);
		if (waitKey(delay) == 27) {
			return;
		}
	}
}

int main(int argc ,char** argv) {
	const char* keys = {
		"{help h	|		| show help message}"
		"{pd 		|		| path of directory contains positive images}"
		"{nd		|		| path of directory contains negative images}"
		"{td		|		| path of directory contains test images}"
		"{tv		|		| test video file name}"
		"{dw 		|		| width of the detector}"
		"{dh		|		| height of the detctor}"
		"{f			| false	| indicates if the program will generate and use mirrored samples or not}"
		"{d 		| false | train twice}"
		"{t			| false	| test a trained detector}"
		"{v			| false	| visualize training steps}"
		"{fn		| my_detector.yml | file name of trained SVM}"
	};
	CommandLineParser parser(argc ,argv ,keys);
	if (parser.has("help")) {
		parser.printMessage();
		exit(0);
	}
	
	<!-- merge -->
	if (pos_dir.empty() || neg_dir.empty()) {
		parser.printMessage();
		cout << "Wrong number of parameters.\n\n"
			<< "Example command line:\n" << argv[0] << " -dw=64 -dh=128 -pd=/INRIAPerson/96X"
	}
}
