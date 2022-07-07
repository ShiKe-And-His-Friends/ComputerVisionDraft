#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "ClibaHelp.h"

using namespace std;
using namespace cv;

void sv_decompse_homography(Mat& H ,const Mat &camera_intrisics ,const Mat &distortion_coefficients ,Mat &rVec ,Mat &tVec) {
	// normal orthogonal 
	double norm = sqrt(H.at<double>(0 ,0)*H.at<double>(0, 0) + H.at<double>(1, 0)*H.at<double>(1, 0) + H.at<double>(2, 0)*H.at<double>(2, 0));
	H /= norm;
	cout << "norm: " << norm << endl;
	cout << "H: " << endl << H << endl;

	Mat rVec1 = H.col(0);
	Mat rVec2 = H.col(1);
	Mat rVec3 = rVec1.cross(rVec2);
	tVec = H.col(2);

	Mat R(3, 3, CV_64F);
	for (int i = 0; i < 3; i++) {
		R.at<double>(i, 0) = rVec1.at<double>(i);
		R.at<double>(i, 1) = rVec2.at<double>(i);
		R.at<double>(i, 2) = rVec3.at<double>(i);
	}
	cout << "R (before decompse): " << endl << R << endl;
	
	// decompose
	Mat W ,U ,Vt;
	SVDecomp(R ,W ,U ,Vt);
	R = U * Vt;
	cout << "R (after decompse): " << endl << R << endl;

	Rodrigues(R, rVec); // rVec tVec

}

void pose_for_homography(Mat &photo1 ,Mat &photo2 , Mat &camera_intrisics ,Mat &distortion_coefficients) {

	//相机尺寸
	Size patternSize(6, 9);
	Size minFindSize(2 ,2);
	
	vector<Point2f> corners1 ,corners2;
	bool found1 = findChessboardCorners(photo1, patternSize, corners1);
	if (found1) {
		cout << "find corner[1]" <<endl;
		cornerSubPix(photo1 ,corners1 , minFindSize ,Size(-1 ,-1) ,TermCriteria(TermCriteria::COUNT | TermCriteria::EPS ,48 ,0.01));
	}
	else {
		cout << "no find corner[1]" << endl;
		return;
	}
	bool found2 = findChessboardCorners(photo2, patternSize, corners2);
	if (found2) {
		cout << "find corner[2]" << endl;
		cornerSubPix(photo2, corners2, minFindSize, Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 48, 0.01));
	}
	else {
		cout << "no find corner[2]" << endl;
		return;
	}

	//画图
	Mat photo_chessboard_found_1 , photo_chessboard_found_2;
	photo_chessboard_found_1 = photo1.clone();
	photo_chessboard_found_2 = photo2.clone();
	drawChessboardCorners(photo_chessboard_found_1,patternSize ,corners1 ,found1);
	drawChessboardCorners(photo_chessboard_found_2, patternSize, corners2 ,found2);
	imshow("corner window 1" , photo_chessboard_found_1);
	waitKey(1000);
	imshow("corner window 2", photo_chessboard_found_2);
	waitKey(1000);
	destroyAllWindows();

	// 平面坐标
	vector<Point2f> planerAxis;
	vector<Point3f> planerAxis_3f;
	ClibaHelp* clibaHelp = new ClibaHelp;
	clibaHelp->calcChessboards(patternSize ,planerAxis_3f);

	vector<Point3f>::iterator it,end;
	it = planerAxis_3f.begin();
	end = planerAxis_3f.end();
	cout << "num " << (end - it) << endl;
	for (; it!=end ; it++) {
		float x = (*it).x;
		float y = (*it).y;
		cout << x << " " << y << endl;
		planerAxis.push_back(Point2f(x, y));
	}

	// undistort 校畸
	Mat image_points_1, image_points_2;
	undistortPoints(corners1, image_points_1, camera_intrisics, distortion_coefficients);
	undistortPoints(corners2, image_points_2, camera_intrisics, distortion_coefficients);

	// homography 单应
	Mat H_photo_1 = findHomography(planerAxis ,image_points_1);
	Mat H_photo_2 = findHomography(planerAxis, image_points_2);
	cout << "H_photo_1" << endl << H_photo_1 << endl;
	cout << "H_photo_2" << endl << H_photo_2 << endl;

	Mat photo1_rVec, photo1_tVec;
	Mat photo2_rVec, photo2_tVec;

	// 可用SVDecomp()方法
	sv_decompse_homography(H_photo_1,camera_intrisics ,distortion_coefficients , photo1_rVec , photo1_tVec);
	sv_decompse_homography(H_photo_2, camera_intrisics, distortion_coefficients , photo2_rVec , photo2_tVec);
	cout << "photo1_rVec " << endl << photo1_rVec << endl;
	cout << "photo1_tVec " << endl << photo1_tVec << endl;
	cout << "photo2_rVec " << endl << photo2_rVec << endl;
	cout << "photo2_tVec " << endl << photo2_tVec << endl;
	
	// 也可用solvePnP()方法
	/*
	solvePnP(planerAxis_3f,image_points_1 ,camera_intrisics ,distortion_coefficients ,photo1_rVec ,photo1_tVec);
	solvePnP(planerAxis_3f, image_points_2, camera_intrisics, distortion_coefficients, photo2_rVec, photo2_tVec);
	cout << "photo1_rVec " << endl << photo1_rVec << endl;
	cout << "photo1_tVec " << endl << photo1_tVec << endl;
	cout << "photo2_rVec " << endl << photo2_rVec << endl;
	cout << "photo2_tVec " << endl << photo2_tVec << endl;
	*/

	// 画图
	drawFrameAxes(photo1 ,camera_intrisics , distortion_coefficients , photo1_rVec, photo1_tVec,2* square_size);
	drawFrameAxes(photo2, camera_intrisics, distortion_coefficients, photo2_rVec, photo2_tVec, 2 * square_size);
	imshow("axes windows 1 " ,photo1);
	waitKey(1000);
	imshow("axes windows 2 ", photo2);
	waitKey(1000);
	destroyAllWindows();

	// 释放
	delete clibaHelp;
}

const char* help_info = 
	"{ help h |h | show message }"
	"{ photo1 | left01.jpg | }"
	"{ photo2 | left09.jpg | }"
	"{ photo3 | left14.jpg | more important photo }"
	"{ intrinsics | left_intrinsics.yml | cmaera intrinsics out file }";

int main(int argc ,char** argv) {

	CommandLineParser parser(argc ,argv , help_info);
	
	// 输入数据
	string photo1_name = parser.get<string>("photo1");
	string photo2_name = parser.get<string>("photo3");
	string intrinsics_file_name = parser.get<string>("intrinsics");
	string photo1_dir = samples::findFile(photo1_name);
	string photo2_dir = samples::findFile(photo2_name);
	string intrisics_file_dir = samples::findFile(intrinsics_file_name);
	
	cout << "photo1_dir " << endl << photo1_dir << endl;
	cout << "photo2_dir " << endl << photo2_dir << endl;
	cout << "intrisics_file_dir " << endl << intrisics_file_dir << endl;

	FileStorage fileStorage(intrisics_file_dir ,FileStorage::READ);
	Mat camera_intrisics, camera_distortion_coefficient;

	// 相机参数
	fileStorage["camera_matrix"] >> camera_intrisics;
	fileStorage["distortion_coefficients"] >> camera_distortion_coefficient;
	cout << "camera_matrix" << endl << camera_intrisics << endl;
	cout << "distortion_coefficients" << endl << camera_distortion_coefficient << endl;

	// 棋盘格图片
	Mat photo1 = imread(photo1_dir ,IMREAD_UNCHANGED);
	Mat photo2 = imread(photo2_dir, IMREAD_UNCHANGED);

	// 用SVDecomp计算单应矩阵
	pose_for_homography(photo1 ,photo2 ,camera_intrisics , camera_distortion_coefficient);


	cout << "" << endl << "" << endl;

	return 0;
}