#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc ,char** argv) {

	string lenaPath = samples::findFile("lena.jpg");
	cout << lenaPath << endl;

	Mat lena = imread(lenaPath, IMREAD_GRAYSCALE);
	Mat lena_Deepcopy_Photo;
	lena.copyTo(lena_Deepcopy_Photo);


	cout << "rows " << lena.rows << " cols " << lena.cols << endl;

	namedWindow("lena");
	imshow("lena", lena_Deepcopy_Photo);
	waitKey(1000);

	return 0;
}
