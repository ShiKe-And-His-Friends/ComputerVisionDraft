/**
 * An example program illustrates the use of cv::findContours
 * and cv::drawContours
 * */
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;


static void help(char** argv) {
	cout <<"\nThis program illustrates the use of findContours and drawContours\n"
		<<"The original iamge is put up along with the image of drawn contours\n"
		<<"UsageL\n";
	cout <<argv[0]
		<<"\nA trackbar is put up which controls the contour level from -3 to 3\n"
		<<endl;
}

const int w = 500;
int levels = 3;

vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

static void on_trackbar(int ,void*) {
	Mat cnt_img = Mat::zeros(w ,w ,CV_8UC3);
	int _levels = levels - 3;
	drawContours(cnt_img ,contours ,_levels <= 0 ? 3: -1 
		,Scalar(128 ,255 ,255) ,3 ,LINE_AA ,hierarchy ,std::abs(_levels));
	imshow("contours" ,cnt_img);
}

int main(int argc ,char** argv) {
	
	return 0;
}
