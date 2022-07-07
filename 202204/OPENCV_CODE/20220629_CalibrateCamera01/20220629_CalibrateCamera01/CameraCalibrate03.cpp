/**
	file:///D:/4.6.0/d9/dab/tutorial_homography.html
	
	decompose_homography
	homography_from_camera_dispalcement
	pose_from_homography
	
*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "ClibaHelp.h"

using namespace std;
using namespace cv;

void checkVectorCalc() {
	//计算旋转矩阵
	Mat origin(3, 1, CV_64F, Scalar(0));
	Mat R1 = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	Mat tVec1 = (Mat_<double>(3, 1) << -0.5, 0.5, 1);
	Mat origin1 = R1 * origin + tVec1;

	// Mat R1_to_R1 = cameraMatix * homograhpy * cameraMatix.inv();
	// Mat tVec1_to_tVec2 = normal1.dot(tVec1);

	cout << "R1: " << endl << R1 << endl;
	cout << "origin: " << endl << origin << endl;
	cout << "origin1: " << endl << origin1 << endl;

	// normal orthogonal quick 快速正交化
	Mat H = (Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
	double index_1_0 = H.at<double>(1, 0);
	double norm = sqrt(H.at<double>(0, 0) *H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0) + H.at<double>(2, 0) * H.at<double>(2, 0));

	cout << "index_1_0 " << endl << index_1_0 << endl;
	cout << "norm" << endl << norm << endl;
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
}

int main3(int argc ,char** agrv) {

	string circle_Photo_Dir1 = "..//CameraData//Img-01.bmp";
	string circle_Photo_Dir2 = "..//CameraData//Img-08.bmp";
	string circle_Photo_Dir3 = "..//CameraData//Img-22.bmp";

	Mat circle_Photo1 = imread(circle_Photo_Dir1, IMREAD_UNCHANGED);
	Mat circle_Photo2 = imread(circle_Photo_Dir2, IMREAD_UNCHANGED);
	Mat circle_Photo3 = imread(circle_Photo_Dir3, IMREAD_UNCHANGED);

	int flags = CALIB_CB_SYMMETRIC_GRID;
	Size patternSize(14,13);
	vector<Point2f> circle_Photo_Corners1;
	vector<Point2f> circle_Photo_Corners2;
	vector<Point2f> circle_Photo_Corners3;
	vector<Point3f> circle_Photo_Axis;

	bool found1 =  findCirclesGrid(circle_Photo1 , patternSize ,circle_Photo_Corners1 ,flags);
	if (found1) {
		cout << "found circle[1]" << endl;
	}
	else {
		cout << "no found circle[1]" << endl;
		return -1;
	}
	bool found2 = findCirclesGrid(circle_Photo2, patternSize, circle_Photo_Corners2, flags);
	if (found2) {
		cout << "found circle[2]" << endl;
	}
	else {
		cout << "no found circle[2]" << endl;
		return -2;
	}
	bool found3 = findCirclesGrid(circle_Photo3, patternSize, circle_Photo_Corners3, flags);
	if (found3) {
		cout << "found circle[3]" << endl;
	}
	else {
		cout << "no found circle[3]" << endl;
		return -3;
	}

	drawChessboardCorners(circle_Photo1 , patternSize ,circle_Photo_Corners1 ,found1);
	drawChessboardCorners(circle_Photo2, patternSize, circle_Photo_Corners2, found2);
	drawChessboardCorners(circle_Photo3, patternSize, circle_Photo_Corners3, found3);

	imshow("cricle photo 1" , circle_Photo1);
	waitKey(1000);
	destroyWindow("cricle photo 1");
	imshow("cricle photo 2", circle_Photo2);
	waitKey(1000);
	destroyWindow("cricle photo 2");
	imshow("cricle photo 3", circle_Photo3);
	waitKey(1000);
	destroyWindow("cricle photo 3");

	ClibaHelp* clibaHelp = new ClibaHelp;
	clibaHelp->calcChessboards(patternSize ,circle_Photo_Axis);

	// 由相机内参、外参，计算空间点的单应矩阵H / PnP计算
	clibaHelp->decomposeMatrix(circle_Photo1, circle_Photo2 ,patternSize ,circle_Photo_Axis ,circle_Photo_Corners1, circle_Photo_Corners2);

	delete clibaHelp;

	return 0;
}