/*
	camera clibration
	*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "ClibaHelp.h"

int main6(int argc ,char** argv) {

	string circle_Photo_Dir1 = "..//CameraData//Img-01.bmp";
	string circle_Photo_Dir2 = "..//CameraData//Img-08.bmp";
	string circle_Photo_Dir3 = "..//CameraData//Img-22.bmp";

	Mat circle_Photo1 = imread(circle_Photo_Dir1, IMREAD_UNCHANGED);
	Mat circle_Photo2 = imread(circle_Photo_Dir2, IMREAD_UNCHANGED);
	Mat circle_Photo3 = imread(circle_Photo_Dir3, IMREAD_UNCHANGED);

	int flags = CALIB_CB_SYMMETRIC_GRID;
	Size patternSize(14, 13);
	Size minSize(5, 5);
	vector<Point2f> circle_Photo_Corners1;
	vector<Point2f> circle_Photo_Corners2;
	vector<Point2f> circle_Photo_Corners3;
	vector<Point3f> circle_Photo_Axis;
	TermCriteria termCriteria(TermCriteria::COUNT | TermCriteria::EPS ,30 ,0.001);

	/*
	cvtColor(circle_Photo1, circle_Photo1, COLOR_BGR2GRAY);
	cvtColor(circle_Photo2, circle_Photo2, COLOR_BGR2GRAY);
	cvtColor(circle_Photo3, circle_Photo3, COLOR_BGR2GRAY);
	*/

	bool found1 = findCirclesGrid(circle_Photo1, patternSize, circle_Photo_Corners1, flags);
	if (found1) {
		cout << "found circle[1]" << endl;
		cornerSubPix(circle_Photo1 ,circle_Photo_Corners1 ,minSize ,Size(-1 ,-1) ,termCriteria);
	}
	else {
		cout << "no found circle[1]" << endl;
		return -1;
	}
	bool found2 = findCirclesGrid(circle_Photo2, patternSize, circle_Photo_Corners2, flags);
	if (found2) {
		cout << "found circle[2]" << endl;
		cornerSubPix(circle_Photo2, circle_Photo_Corners2, minSize, Size(-1, -1), termCriteria);
	}
	else {
		cout << "no found circle[2]" << endl;
		return -2;
	}
	bool found3 = findCirclesGrid(circle_Photo3, patternSize, circle_Photo_Corners3, flags);
	if (found3) {
		cout << "found circle[3]" << endl;
		cornerSubPix(circle_Photo3, circle_Photo_Corners3, minSize, Size(-1, -1), termCriteria);
	}
	else {
		cout << "no found circle[3]" << endl;
		return -3;
	}

	drawChessboardCorners(circle_Photo1, patternSize, circle_Photo_Corners1, found1);
	drawChessboardCorners(circle_Photo2, patternSize, circle_Photo_Corners2, found2);
	drawChessboardCorners(circle_Photo3, patternSize, circle_Photo_Corners3, found3);

	imshow("cricle photo 1", circle_Photo1);
	waitKey(1000);
	destroyWindow("cricle photo 1");
	imshow("cricle photo 2", circle_Photo2);
	waitKey(1000);
	destroyWindow("cricle photo 2");
	imshow("cricle photo 3", circle_Photo3);
	waitKey(1000);
	destroyWindow("cricle photo 3");

	ClibaHelp* clibaHelp = new ClibaHelp;
	clibaHelp->calcChessboards(patternSize, circle_Photo_Axis);

	// 计算相机内参、外参
	Mat cameraMatrix, distCoeffs;

	vector<Mat> mats;
	vector<vector<Point2f>> cornerss;
	mats.push_back(circle_Photo1);
	mats.push_back(circle_Photo2);
	mats.push_back(circle_Photo3);
	cornerss.push_back(circle_Photo_Corners1);
	cornerss.push_back(circle_Photo_Corners2);
	cornerss.push_back(circle_Photo_Corners3);

	clibaHelp->runClibration(mats ,cornerss ,patternSize ,cameraMatrix , distCoeffs);

	delete clibaHelp;

	return 0;
}