/*
	camera clibration all photos
	*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "ClibaHelp.h"

int main(int argc, char** argv) {

	// 读取文件夹下的所有孔洞图片的目录
	ClibaHelp* clibaHelp = new ClibaHelp;
	vector<String> photos_dirs;
	string circle_Photo_Dir1 = "..//CameraData//Ignore_images";
	
	clibaHelp->getAllFileFromDirctory(circle_Photo_Dir1 ,photos_dirs ,0);

	// 读取所有孔洞图片
	vector<Mat> photos;
	vector<String>::iterator it_photos_dirs, end_photos_dirs;
	it_photos_dirs = photos_dirs.begin();
	end_photos_dirs = photos_dirs.end();

	for (; it_photos_dirs != end_photos_dirs; it_photos_dirs++) {
		Mat circle_Photo = imread((*it_photos_dirs), IMREAD_UNCHANGED);
		photos.push_back(circle_Photo);
	}
	cout << "Photos Size " << photos.size() << endl;

	// 找图的特征角点
	int return_flag_values = 0;
	int flags = CALIB_CB_SYMMETRIC_GRID;
	Size patternSize(14, 13);
	Size minSize(5, 5);
	vector<vector<Point2f>> circle_Photo_Corners;
	vector<Point3f> circle_Photo_Axis;
	TermCriteria termCriteria(TermCriteria::COUNT | TermCriteria::EPS, 0, 0.001);

	vector<Mat>::iterator it_photos, end_photos;
	it_photos = photos.begin();
	end_photos = photos.end();

	for (; it_photos != end_photos; it_photos++) {
		return_flag_values ++;

		vector<Point2f> circle_Photo_Corner_Sub;
		bool found = findCirclesGrid((*it_photos), patternSize ,circle_Photo_Corner_Sub ,flags);
		if (found) {
			cout << "found circle[1]" << endl;
			cornerSubPix((*it_photos), circle_Photo_Corner_Sub, minSize, Size(-1, -1), termCriteria);
		}
		else {
			cout << "no found circle[1]" << endl;
			return -return_flag_values;
		}
		circle_Photo_Corners.push_back(circle_Photo_Corner_Sub);

		// 画图看一下标定效果
		drawChessboardCorners((*it_photos), patternSize, circle_Photo_Corner_Sub, found);
		imshow("cricle photo", (*it_photos));
		waitKey(800);
		destroyWindow("cricle photo");
	}

	/**
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

	clibaHelp->runClibration(mats, cornerss, patternSize, cameraMatrix, cameraMatrix);
	**/

	delete clibaHelp;


	return 0;
}