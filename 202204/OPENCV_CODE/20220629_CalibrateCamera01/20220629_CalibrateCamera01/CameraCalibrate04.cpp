#include <iostream>
#include <opencv2/core.hpp>


using namespace std;
using namespace cv;

int main(int argc ,char** argv) {

	Mat origin(3, 1, CV_64F, Scalar(0));
	Mat R1  = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	Mat tVec1 = (Mat_<double>(3, 1) << -0.5, 0.5, 1);
	Mat origin1 = R1 * origin + tVec1;

	cout << "R1: " << endl << R1 << endl;
	cout << "origin: " << endl << origin << endl;
	cout << "origin1: " << endl << origin1 << endl;

	return 0;
}