#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main2(int argc ,char** argv) {

	string lenaPath = samples::findFile("lena.jpg");
	cout << lenaPath << endl;
	Mat lena = imread(lenaPath, IMREAD_COLOR);
	cout << "rows " << lena.rows << endl;
	cout << "cols" << lena.cols << endl;

	namedWindow("lena");
	imshow("lena", lena);
	waitKey(1000);

	return 0;
}
