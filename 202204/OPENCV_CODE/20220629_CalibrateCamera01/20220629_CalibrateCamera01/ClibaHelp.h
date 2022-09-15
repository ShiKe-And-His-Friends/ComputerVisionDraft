#ifndef  CLlIB_HELP_H
#define CLlIB_HELP_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>


using namespace std;
using namespace cv;

const float square_size = 0.025; // ?

class ClibaHelp
{

private:
	const string homograph_yaml_dir = "..//CameraData//20220707_homography_mat_information.yaml";
	
	void homographyInfoClean();
	void homographyInfoSave(const Mat &r, const Mat &t, const Mat& n ,int num = 1);
		

public :

	const static bool DEBUG_SWITCH = true;

	Scalar randomColor(RNG &rng);

	void calcChessboards(const Size &chessboardSize, vector<Point3f> &corners);

	void decomposeMatrix(const Mat &mat1, const Mat &mat2, Size &patternSize
		,vector<Point3f> &axis, vector<Point2f> &corners1, vector<Point2f> &corners2);

	void runClibration(vector<Mat> &mats, vector<vector<Point2f>> &cornerss, Size size, Mat &cameraMatrix, Mat &distCoeffs);

	int getAllFileFromDirctory(string path ,vector<String> &result,int needNum);

};

#endif