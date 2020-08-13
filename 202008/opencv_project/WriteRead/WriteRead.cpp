#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc ,char ** argv) {
	Mat img;
	img = imread("../../lenna.jpg");
	if (img.empty()) {

		cout << "" << endl;
		return -1;
	}
	imshow("Opencv Picture" ,img);
	waitKey(0);
	return 0;
}
