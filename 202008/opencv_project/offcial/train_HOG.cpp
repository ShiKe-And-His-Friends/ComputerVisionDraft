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