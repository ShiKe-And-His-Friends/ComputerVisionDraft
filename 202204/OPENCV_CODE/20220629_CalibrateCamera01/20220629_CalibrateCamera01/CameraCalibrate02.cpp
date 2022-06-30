#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc ,char** argv) {

	string lenaPath = samples::findFile("lena.jpg");
	Mat lena = imread( lenaPath, IMREAD_COLOR); 

	imshow("" ,lena);

	return 0;
}
