#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>


using namespace std;
using namespace cv;

int main(int argc ,char** argv) {

	// normal orthogonal quick 快速正交化
	Mat H  = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	double index_1_0 = H.at<double>(1,0);
	double norm = sqrt(H.at<double>(0 ,0) *H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0) + H.at<double>(2, 0) * H.at<double>(2, 0) );
	
	cout << "index_1_0 " << endl << index_1_0 << endl;
	cout << "norm" << endl  << norm << endl;
	cout << "H" << endl << H << endl;

	H /= norm;
	Mat c1 = H.col(0);
	Mat c2 = H.col(1);
	Mat c3_1 = c1.cross(c2);
	Mat c3_2 = c2.cross(c1);
	Mat tvec = H.col(2);
	cout << "H" << endl << H << endl;
	cout << "c1" << endl << c1 << endl;
	cout << "c2" << endl << c2 << endl;
	cout << "tvec" << endl << tvec << endl;
	cout << "c3_1" << endl << c3_1 << endl;
	cout << "c3_2" << endl << c3_2 << endl;

	Mat R(3, 3, CV_64F);
	for (int i = 0; i < 3; i++) {
		R.at<double>(i, 0) = c1.at<double>(i); 
		R.at<double>(i, 1) = c2.at<double>(i);
		R.at<double>(i, 2) = c3_2.at<double>(i);
	}
	cout << "R" << endl << R << endl;
	cout << "" << endl << "" << endl;

	return 0;
}