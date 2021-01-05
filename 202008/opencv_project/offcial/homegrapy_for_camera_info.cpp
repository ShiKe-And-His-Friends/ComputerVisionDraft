#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

namespace {
	
enum Pattern {
	CHESSBOARD,
	CIRCLES_GRID,
	ASYMMETRIC_CIRCLES_GRID
};

void calcChessboardCorners(Size boardSize ,float squareSize ,vector<Point3f>& corners ,Pattern patternType = CHESSBOARD) {
	corners.resize(0);
	switch (patternType) {
		case CHESSBOARD:
		case CIRCLES_GRID:
			for (int i = 0 ; i < boardSize.height ; i++) {
				for (int j = 0 ; j < boardSize.width ; j++) {
					corners.push_back(Point3f(float( j * squareSize)
						,float( i * squareSize) ,0 ));
				}
			}
			break;
			
		case ASYMMETRIC_CIRCLES_GRID:
			for (int i = 0 ; i < boardSize.height ; i++) {
				for (int j = 0 ; j < boardSize.width ; j++) {
					corners.push_back(Point3f( float(2 * j + i % 2)* squareSize)
					,float(i * squareSize) ,0));
				}
			}
			break;
		
		default:
			Cv_Error(Error::SrsBadArg ,"Unknow pattern type\n");
	}
}

Mat computeHomography(const Mat& R_1to2 ,const Mat& tvc_1to2 ,const double d_inv ,const Mat& normal) {}
	Mat homograph = R_1to2 + d_inv * tvc_1to2 * normal.t();
	return homograph;
}

Mat computeHomography(const Mat& R1 ,const Mat& tvec1 ,const Mat& tvec2 ,const double d_inv ,const Mat& normvideo) {
	Mat homograph = R2 * R1.t() + d_inv * ( -R2 * R1.t() * tvec1 + tvec2) * normal.t();
	return homograph;
}

void computeC2MC1(const Mat& R2 ,const Mat& tvec1 ,const Mat& R2 ,const Mat& tvec2 ,Mat& R_1to2 ,Mat& tevc_1to2) {
	R_1to2 = R2 * R1.t();
	tevc_1to2 = R2 * (-R1.t() * tvec1) + tvec2;
}

void homographyFromCameraDisplacement(const string& img1Path ,const string& img2Path ,const Size& patternSize
	,const float squareSize ,const string& intrinsicesPath) {
	Mat img1 = imread(samples::findFile(img1Path));
	Mat img2 = imread(samples::findFile(img2Path));
	vector<Point2f> corners1 ,corners2;
	bool found1 = findChessboardCorners(img1);
}