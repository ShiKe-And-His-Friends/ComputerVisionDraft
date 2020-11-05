#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

static void help () {
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n"<<
		"\nHot keys:\n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\t left mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button -set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set GC_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set GC_PR_FGD pixels\n\n" << endl;
}

const Scalar RED = Scalar(0 ,255 ,255);
const Scalar PINK = Scalar(230 ,130 ,255);
const Scalar BLUE = Scalar(255 ,0 ,0);
const Scalar LIGHTBLUE = Scalar(255 ,255 ,160);
const Scalar GREEN = Scalar(0 ,255 ,0):

const int BGD_KEY = EVENT_FLAG_CTRLKEY;
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;

static void getBinMask (const Mat& comMask ,Mat& binMask) {
	if (comMask.empty() || comMask.type() != CV_8UC1) {
		CV_Error(Error::StsBadArg ,"comMask is empty ot has incorrect type (not CV_8UC1)");
	}
	if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols) {
		binMask.create(comMask.size() ,CV_8UC1);
	}
	binMask = comMask & 1;
}


