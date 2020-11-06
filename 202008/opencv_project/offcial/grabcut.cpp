#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

static void help () {
	cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabCut will attempt to segment it out.\n"
		"Call:\n"
		"./grabCut <image_name>\n"
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

class GCApplication {
public:
	enum {
		NOT_SET = 0 ,
		IN_PROCESS = 1 ,
		SET = 2 
	};

	static const int radius = 2;
	static const int thickness = 1;

	void reset();
	void setImageAndWinName(const Mat& _iamge ,const string& _winName);
	void showImage() const;
	void mouseClick(int event ,int x ,int y ,int flags ,void* param);
	int nextIter();
	
	int getIterCount() const {
		return iterCount;
	}

private:
	void setRectInMask();
	void setLblsInMask(int flags ,Point p ,bool isPr);

	const string* winName;
	const Mat* image;
	Mat mask;
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblesState;
	bool isInitialized;

	Rect rect;
	vector<Point> fgbPxls, bgdPxls, prFgdPxels, prBgPxls;
	int iterCount;
};

void GCApplication::reset() {
	if (!mask.empty()) {
		mask.setTo(Scalar::all(GC_BGD));
	}
	bgdPxls.clear();
	fgbPxls.clear();
	prBgdPxls.clear();
	prFgdpxls.clear();

	isInitialized = false;
	rectState = NOT_SET;
	bkbsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

void GCApplication::setLblsInMask(int flags ,Point p ,bool isPr) {
	
	vector<Point> *bpxls ,*fpxls;
	uchar bvalue ,fvalue;
	if (!isPr) {
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;
		fvalue = GC_PR_FGD;
	}
	if (flags & BGD_KEY) {
		bpxls->push_back(p);
		circle(mask ,p ,radius ,bvalue ,thickness);
	}
	if (flags & FGD_KEY) {
		fpxls->push_back(p);
		circle(mask ,p ,radius ,fvalue ,thickness);
	}
}

void GCApplication::mouseClick(int event ,int x ,int y ,int flags ,void*) {
	//TODO add bad args check
	switch (event) {
		case EVENT_LBUTTONDOWN:
			bool isb = (flags & BGD_KEY) != 0;
			bool isf = (flags & FGD_KEY) != 0;
			if (rectState == NOT_SET && !isb && !isf) {
				rectState = IN_PROCESS;
				rect = Rect(x ,y ,1 ,1);
			}
			if ((isb || isf) && rectState == SET) {
				lblsState == IN_PROCESS;
			}
			break;
			
		case EVENT_RBUTTONDOWN:
			// set GC_PR_BGD labels
			bool isb = (flags & BGD_KEY) != 0;
			bool isf = (flags & FGD_KEY) != 0;
			if ((isb || isf) && rectState == SET) {
				prLblesState = IN_PROCESS;
			}
			break;
			
		case EVENT_LBUTTONUP:
			if (rectState == IN_PROCESS) {
				rect = Rect(Point(rect.x ,rect.y) ,Point(x ,y));
				rectState = SET;
				setRectInMask();
				CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgPxls.empty() && prBgPxls.empty());
				showImage();
			}
			if (lblsState == IN_PROCESS) {
				setLblsInMask(flags ,Point(x ,y) ,false);
				lblsState = SET;
				showImage();
			}
			break;
			
		case EVENT_RBUTTONUP:
			if (prLblesState == IN_PROCESS) {
				setLblsInMask(flags ,Point(x ,y) ,true);
				prLblesState = SET;
				showImage();
			}
			break;
			
		case EVENT_MOUSEMOVE:
			if (rectState == IN_PROCESS) {
				rect = Rect(Point(rect.x ,rect.y) ,Point(x ,y));
				CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxels.empty());
				showImage();
			} else if (lblsState == IN_PROCESS) {
				setLblsInMask(flags ,Point(x ,y) ,false);
				showImage();
			} else if (prLblesState == IN_PROCESS) {
				setLblsInMask(flags ,Point(x ,y) ,true);
				showImage();
			}
			break;
	}
}

int GCApplication::nextIter() {
	if (isInitialized) {
		grabCut(*image ,mask ,recr ,bgdModel ,fgdModel ,1);
	} else {
		if (rectState != SET) {
			return iterCount;
		}
		if (lblsState == SET || prLblesState == SET) {
			grabCut(*image ,mask ,rect ,bgdModel ,fgdModel ,1 ,GC_INIT_WITH_MASK);
		} else {
			grabCut(*image ,mask ,rect ,bgdModel ,fgdModel ,1 ,GC_INIT_WITH_RECT);
		}
		isInitialized = true;
	}
	iterCount ++;
	bgdPxls.clear();
	fgdPxls.clear();
	prBgPxls.clear();
	prFgdPxels.clear();
	
	return iterCount;
}

GCApplication gcapp;

static void on_mouse(int event ,int x ,int y ,int flags ,void* param) {
	gcapp.mouseClick(event ,x ,y ,flags ,param);
	
}
