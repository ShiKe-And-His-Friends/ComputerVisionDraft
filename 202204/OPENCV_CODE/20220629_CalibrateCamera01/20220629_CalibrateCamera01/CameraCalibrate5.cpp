#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

int main(int argc ,char** agrv) {

	string circle_Photo_Dir1 = "..//CameraData//Img-01.bmp";
	string circle_Photo_Dir2 = "..//CameraData//Img-01.bmp";
	string circle_Photo_Dir3 = "..//CameraData//Img-01.bmp";

	Mat circle_Photo1 = imread(circle_Photo_Dir1, IMREAD_UNCHANGED);
	Mat circle_Photo2 = imread(circle_Photo_Dir2, IMREAD_UNCHANGED);
	Mat circle_Photo3 = imread(circle_Photo_Dir3, IMREAD_UNCHANGED);

	int flags = CALIB_CB_SYMMETRIC_GRID;
	Size patternSize(12,12);
	vector<Point2f> circle_Photo_Corners1;
	vector<Point2f> circle_Photo_Corners2;
	vector<Point2f> circle_Photo_Corners3;

	bool found1 =  findCirclesGrid(circle_Photo1 , patternSize ,circle_Photo_Corners1 ,flags);

	if (found1) {
		cout << "found circle[1]" << endl;
	}
	else {
		cout << "no found circle[1]" << endl;
	}

	imshow("cricle photo 1" , circle_Photo1);
	waitKey(1000);
	imshow("cricle photo 2", circle_Photo2);
	waitKey(1000);
	imshow("cricle photo 2", circle_Photo2);
	waitKey(1000);

	return 0;
}