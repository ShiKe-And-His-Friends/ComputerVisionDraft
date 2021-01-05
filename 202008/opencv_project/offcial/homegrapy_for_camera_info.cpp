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
	bool found1 = findChessboardCorners(img1 ,patternSize ,corners1);
	bool found2 = findChessboardCorners(img2 ,patternSize ,corners2);
	
	if (!found1 || !found2) {
		cout << "Error ,cannot find the chessboard corners in both images." << endl;
		return;
	}
	
	vector<Point3f> objectPoints;
	calcChessboardCorners(patternSize ,squareSize ,objectPoints);
	
	FileStorage fs(samples::findFile(intrinsicesPath) ,FileStorage::READ);
	Mat cameraMatrix ,distCoeffs;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	
	Mat rvec1 ,tvec1;
	solvePnP(objectPoints ,corners1 ,cameraMatrix ,distCoeffs ,rvec1 ,tvec1);
	Mat rvec1 ,tvec2;
	solvePnP(objectPoints ,corners2 ,cameraMatrix ,distCoeffs ,rvec2 ,tvec2);
	
	Mat img1_copy_pose = img1.clone();
	Mat img2_copy_pose = img2.clone();
	Mat img_draw_poses;
	drawFrameAxes(img1_copy_pose ,cameraMatrix ,distCoeffs ,rvec1 ,tvec1 ,2 * squareSize);
	drawFrameAxes(img2_copy_pose ,cameraMatrix ,distCoeffs ,rvec2 ,tvec2 , 2 * squareSize);
	hconcat(img1_copy_pose ,img2_copy_pose ,img_draw_poses);
	imshow("Chessboard poses" ,img_draw_poses);
	
	Mat R1 ,R2;
	Rodrigues(rvec1 ,R1);
	Rodrigues(rvec2 ,R2);
	
	Mat R_1to2 ,t_1to2;
	computeC2MC1(R1 ,tvec1 ,R2 ,tvec2 ,R_1to2 ,t_1to2);
	Mat rvec_1to2;
	Rodrigues(R_1to2 ,rvec_1to2);
	
	Mat normal = (Mat_<double>(3 ,1) << 0 ,1 ,1);
	Mat normal1 = R1 * normal;
	
	Mat origin(3 ,1 ,CV_64F ,Scalar(0));
	Mat origin1 = R1 * origin + tvec1;
	double d_inv1 = 1.0 / normal1.dot(origin1);
	
	Mat homograph_euclidean = computeHomography(R_1to2 ,t_1to2 ,d_inv ,normal1);
	Mat homography = cameraMatrix * homograph_euclidean * cameraMatrix.inv();
	
	homograph /= homograph.at<double>(2 ,2);
	homograph_euclidean /= homograph_euclidean.at<double>(2 ,2);
	
	// same but using absolute camera poses instead of camera displacement ,just for check
	Mat homograph_euclidean2 = computeHomography(R1 ,tvec1 ,R2 ,tvec2 ,d_inv1 ,normal1);
	Mat homograph2 = cameraMatrix * homograph_euclidean2 * cameraMatrix.inv();
	

}