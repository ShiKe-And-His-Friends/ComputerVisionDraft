#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>

using namespace std;
using namespace cv;

int maskSize0 = DIST_MASK_5;
int voronoiType = -1;
int edgeThresh = 100;
int distType0 = DIST_L1;

// The output and temporary images
Mat gray;

// Threshold trackbar callback
static void onTrackbar (int ,void*) {
	static const Scalar colors[] = {
		Scalar(0 ,0 ,0);
		Scalar(255 ,0 ,0);
		Scalar(255 ,128 ,0);
		Scalar(255 ,255 ,0);
		Scalar(0 ,255 ,0);
		Scalar(0 ,128 ,255);
		Scalar(0 ,255 ,255);
		Scalar(0 ,0 ,255);
		Scalar(255 ,0 ,255);
	}

	int maskSize = voronoiType >= 0 ? DIST_MASK_5 : maskSize0;

}

