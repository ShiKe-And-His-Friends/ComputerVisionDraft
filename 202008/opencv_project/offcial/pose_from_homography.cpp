#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

namespace {
	
enum Pattern {
	CHESSBOADRD,
	CIRCLES_GRID,
	ASYMMETRIC_CIRCLES_GRID
};

void calcChessboardCorners(Size boardSize ,float squareSize ,vector<Point3f>& corners 
	,Pattern patternType = CHESSBOADRD) {
	corners.resize(0);
	switch(patternType) {
		case CHESSBOADRD:
		case CIRCLES_GRID:
			for(int i = 0 ; i < boardSize.height ;i++) {
				for(int j = 0 ; j < boardSize.width ; j++) {
					corners.push_back(Point3f( float(j * squareSize) ,float(i * squareSize) ,0));
				}
			}
			break;
			
		case ASYMMETRIC_CIRCLES_GRID:
			for (int i = 0 ; i < boardSize.height ; i++) {
				for (int j = 0 ; j < boardSize.width ; j++) {
					corners.push_back(Point3f(float(2*j + i%2) * squareSize) ,float(i * squareSize) ,0);
				}
			}
			break;
		default:
			CV_Error(Error::StsBadArg ,"Unknown pattern type\n");
	}
}

void poseEstimationFromCoplanarPoints(const string& imgPath ,const string& intrinsicsPath 
	,const size& patternSize) {
	Mat img = imread(samples::findFile(imgPath));
	Mat img_corners = img.clone(), img_pos = img.clone();
	vector<Point2f> corners;
	bool found = findChessboardCorners(img ,patternSize ,corners);
	if (!found) {
		cout << "Cannot find chessboard corners." << endl;
		return;
	}
	drawChessboardCorners(img_corners ,patternSize ,corners ,found);
	imshow("Chessboard corners detection" ,img_corners);
	vector<Point3f> objectPoints;
	drawChessboardCorners(patternSize ,squareSize ,objectPoints);
	vector<Point2f> objectPointPlanar;
	for(size_t i = 0 ; i < objectPoint.size() ; i++) {
		objectPointPlanar.push_back(Point2f(objectPoint[i].x ,objectPoint[i].y));
	}
	FileStorage fs(samples::findFile(intrinsicsPath) ,FileStorage::READ);
	Mat cameraMatrix ,distCoeffs;
	fs["camera_matrix"] >> cameraMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	vector<Point2f> imagePoints;
	undistortPoints(corners ,imagePoints ,cameraMatrix ,distCoeffs);
	Mat H = findHomography(objectPointPlanar ,imagePoints);
	cout << "H:\n" << H << endl;
	double norm = sqrt(H.at<double>(0 ,0) * H.at<double>(0 ,0) +
		H.at<double>(1 ,0) * H.at<double>(1 ,0) + 
		H.at<double>(2 ,0) * H.at<double>(2 ,0));
	H /= norm;
	Mat c1 = H.col(0);
	Mat c2 = H.col(1);
	Mat c3 = c1.cross(c2);
	Mat tvec = H.col(2);
	Mat R(3 ,3 ,CV_64F);
	for (int i = 0 ; i < 3 ; i++) {
		R.at<double>(i ,0) = c1.at<double>(i ,0);
		R.at<double>(i ,1) = c2.at<double>(i ,0);
		R.at<double>(i ,2) = c3.at<double>(i ,0);
	}
	cout << "R (before polar decpmposition):\n" << R << "\ndet(R)" << determinant(R) << endl;
	Mat W ,U ,Vt;
	SVDecomp(R ,W ,U ,Vt);
	R = U * Vt;
	cout << "R (after polar decomposition):\n" << R << "\ndet(R):" << determinant(R) << endl;
	Mat rvec;
	Rodrigues(R ,rvec);
	drawFrameAxes(img_pose ,cameraMatrix ,distCoeffs ,rvec ,tvec ,2 * squareSize);
	imshow("Pose from coplanar points" ,img_pose);
	waitKey();
}

const char* params = 
	"{ help h		|						| print usage}"
	"{ imagePoints	| left04.jpg			| path to a chessboard image}"
	"{ intrinsice	| left_intrinsices.yml	| path to camera intrinsics}"
	"{ width bw		| 9						| chessboard width}"
	"{ height bh	| 6						| chessboard height}"
	"{ square_size	| 0.025					| chessboard square size}";

}

int main (int argc ,char* argv[]) {
	CommandLineParser parser(argc ,argv ,params);
	if (parser.has("help")) {
		parser.about("Code for homography tutorial.\n"
			"Example 1: pose from homography with coplanar points.\n");
		parser.printMessage();
		return 0;
	}
	Size patternSize(parser.get<int>("width") ,parser.get<int>("height"));
	float squareSize = (float)parser.get<double>("square_size");
	poseEstimationFromCoplanarPoints(parser.get<String>("image"),
									parser.get<String>("intrinsice"),
									patternSize ,squareSize);
}