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

using namespace std;
using namespace cv;

const static float square_size = 0.025; // ?

// 棋盘格坐标轴的描点
void calcChessboards(const Size &chessboardSize ,vector<Point3f> &corners) {
	corners.resize(0);
	for (int i = 0 ;i < chessboardSize.width; i++) {
		for (int j = 0; j < chessboardSize.height ; j++) {
			float x = (float)(i*square_size);
			float y = (float)(j*square_size);
			corners.push_back(
				Point3f(x ,y ,0)
			);
		}
	}
	vector<Point3f>::iterator it, end;
	it = corners.begin();
	end = corners.end();
	cout << "Chessboard aixs number :" << (end - it) << endl;

	for (; it != end; it++) {
		cout << "Chessboard aixs x" << (*it).x << " y " << (*it).y << endl;
	}
}

// 分解矩阵
void decomposeMatrix(const Mat &mat1 , const Mat &mat2 ,const Mat &mat3 ,Size &patternSize 
	,vector<Point3f> &corners1, vector<Point3f> &corners2, vector<Point3f> &corners3
	,vector<Point2f> &h_input1 , vector<Point2f> &h_input2) {
	String intrinsicsPath = samples::findFile("left_intrinsics.yml");
	cout << "Camera samples intrinsics path: " << intrinsicsPath << endl;
	FileStorage file(intrinsicsPath ,FileStorage::READ);

	// 读取示例的相机参数
	Mat cameraIntrinsicsMatrix ,cameraDistCoffes;
	file["camera_matrix"] >> cameraIntrinsicsMatrix;
	file["distortion_coefficients"] >> cameraDistCoffes;
	file.release();
	cout << "Intrinsics " << endl << cameraIntrinsicsMatrix << endl;
	cout << "Dist Coffes" << endl << cameraDistCoffes << endl;

	// 用已知点求空间点的P3P计算
	Mat rVecs_1, tVecs_1;
	solvePnP(corners1 ,h_input1 ,cameraIntrinsicsMatrix ,cameraDistCoffes , rVecs_1, tVecs_1);
	cout << "rVecs_1 " << endl << rVecs_1 << endl;
	cout << "tVecs_1" << endl << tVecs_1 << endl;
	cout << "h_input1 " << endl << h_input1 << endl;



}

int main(int argc ,char** agrv) {

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
	vector<Point3f> circle_Photo_Corners_3f_1;
	vector<Point3f> circle_Photo_Corners_3f_2;
	vector<Point3f> circle_Photo_Corners_3f_3;

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

	calcChessboards(patternSize ,circle_Photo_Corners_3f_1);
	calcChessboards(patternSize, circle_Photo_Corners_3f_2);
	calcChessboards(patternSize, circle_Photo_Corners_3f_3);

	// 由相机内参、外参，计算空间点的单应矩阵H / PnP计算
	vector<Point2f> homography_solve_input1;
	vector<Point2f> homography_solve_input2;

	decomposeMatrix(circle_Photo1 ,circle_Photo2 ,circle_Photo3 ,patternSize ,circle_Photo_Corners_3f_1 , circle_Photo_Corners_3f_2, circle_Photo_Corners_3f_3 ,homography_solve_input1 ,homography_solve_input2);

	return 0;
}