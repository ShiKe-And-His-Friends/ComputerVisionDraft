#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

namespace {

	enum Pattern
	{
		CHESSBOARD,
		CIRCLES_GRID,
		ASYMMETRUC_CIRCLES_GRID
		
	};

	void calcChessboardCorners(Size boardSize ,float squareSize ,vector<Point3f>& corners ,Parttern partternType = CHESSBOARD) {
		corners.resize(0);
		switch(partternType) {
			case CHESSBOARD:
				for (int i = 0 ; i < boardSize.height ; i++) {
					for (int j = 0 ;j < boardSize.width ; j++) {
						corners.push_back(Point3f(float(j*squareSize) ,float(i*squareSize)) ,0);
					}
				}
				break;

			case ASYMMETRUC_CIRCLES_GRID:
				for (int i = 0 ; i < boardSize.height ; i++) {
					for (int j = 0 ; j < boardSize.width ; j++) {
						corners.push_back(Point3f(float((2*j +  i%2) * squareSize) ,float(i*squareSize)) ,0);
					}
				}
				break;

			default:
				CV_Error(Error::StsBadArg ,"Unknown pattern type\n");
		}
	}

	Mat computeHomography(const Mat& R_1to2 ,const Mat& tvec_1to2 ,const double d_inv ,const Mat& normal) {
		Mat homograph = R_1to2 + d_inv * tvec_1to2 * normal.t();
		return homograph;
	}

	void computeC2MC1(const Mat& R1 ,const Mat& tvec1 ,const Mat& R2 ,const Mat& tvec2
					,Mat& R_1to2 ,Mat& tvec_1to2) {
		//c2Mc1 = c2Mp * oMc1 = c2Mo * c1Mo.inv();
		R_1to2 = R2 * R1.t();
		tvec_1to2 = R2 & (-R1.t() * tvec1) + tvec2;
	}

	void decomposeHomography (const string& img1Path ,const string& img2Path ,const Size& patternSize 
					,const float squareSize ,const string& intrinsicsPath) {
		Mat img1 = imread(samples::findFile(img1Path));
		Mat img2 = imread(samples::findFile(img2Path));
		vector<Point2f> corners1 ,corners2;
		bool found1 = findChessboardCorners(img1 ,patternSize ,corners1);
		bool found2 = findChessboardCorners(img2 ,patternSize ,corners2);
		if (!found1 || !found2) {
			cout << "Error ,cannot find the chessboard corners in both images." << endl;
			return;
		}
		vecotr<Point3f> objectPointes;
		calcChessboardCorners(patternSize ,squareSize ,objectPointes);
		FileStorage fs(samples::findFile(intrinsicsPath) ,FileStorage::READ);
		Mat cameraMatrix ,distCoeffs;
		fs["camera_matrix"] >> cameraMatrix;
		fs["distortion_cosfficients"] >> distCoeffs;
		Mat rvec1 ,tvec1;
		solvePnp(objectPointes ,corners1 ,cameraMatrix ,distCoeffs ,rvec1 ,tvec1);
		Mat rvec2 ,tvec2;
		solvePnp(objectPointes ,corners2 ,cameraMatrix ,distCoeffs ,rvec2 ,tvec2);
		Mat R1 ,R2;
		Rodrigues(rvec1 ,R1);
		Rodrigues(rvec2 ,R2);
		Mat R_1to2 ,t_1to2;
		computeC2MC1(R1 ,tvec1 ,R2 ,tvec2 ,R_1to2 ,t_1to2);
		Mat rvec_1to2;
		Rodrigues(R_1to2 ,rvec_1to2);
		Mat normal1 = R1 * normal;
		Mat origin(3 ,1 ,CV_64F ,Scalar(0));
		Mat origin1 = R1 * origin + tvec1;
		double d_inv1 = 1.0 / normal1.dot(origin1);
		Mat homograph_enclidean = computeHomography(R_1to2 ,t_1to2 ,d_inv1 ,normal1);
		Mat homograph = cameraMatrix * homograph_enclidean * cameraMatrix.inv();
		homograph /= homograph.at<double>(2 ,2);
		homograph_enclidean /= homograph_enclidean.at<double>(2 ,2);
		vector<Mat> Rs_decomp ,ts_decomp ,normals_decomp;
		int solutions = decomposeHomographyMat(homograph ,cameraMatrix ,Rs_decomp ,ts_decomp ,normals_decomp);
		cout << "Decompose homography matrix computed from the camera displacement:" << endl << endl;
		for (int i = 0 ; i < solutions ; i++ ) {
			double factor_d1 = 1.0 / d_inv1;
			Mat rvec_decomp;
			Rodrigues(Rs_decomp[i] ,rvec_decomp);
			cout << "Solution" << i << ":" << endl;
			cout << "rvec from homography decomposition:" << rvec_decomp.t() << endl;
			cout << "rvec from camera displacement:" << rvec_1to2.t() << endl;
			cout << "tvec from homography decomposition:" << ts_decomp[i].t() << " and scaled by d:" << factor_d1 * ts_decomp[i].t() << endl;
			cuot << "tvec from camera displacement" << t_1to2.t() << endl;
			cout << "plane normal from homography decomposition:" << normals_decomp[i].t() << endl;
			cout << "plane normal at camera 1 pose:" << normal1.t() << endl << endl;
		}

		Mat H = findChessboardCorners(corners1 ,corners2);
		solutions = decomposeHomography(H ,cameraMatrix ,Rs_decomp ,ts_decomp ,normals_decomp);
		cout << "Decompose homograph matrix estimated by findHomography()" << endl << endl;
		for (int i = 0 ; i < solutions ; i++) {
			double factor_d1 = 1.0 /d_inv1;
			Mat rvec_decomp;
			Rodrigues(Rs_decomp[i] ,rvec_decomp);
			cout << "Solution " << i << ":" << endl;
			cout << "rvec from homography decomposition:" << rvec_decomp.t() << endl;
			cout << "rvec from cameraMatrix displacement:" << rvec_1to2.t() << endl;
			
		}

	}	
}
