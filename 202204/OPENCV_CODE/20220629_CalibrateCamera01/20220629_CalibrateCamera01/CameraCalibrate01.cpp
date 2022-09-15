#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "ClibaHelp.h"

using namespace std;
using namespace cv;

const char* help = 
	"{ help h | | need help }"
	"{ image1 | left01.jpg | image check }"
	"{ image2 | left03.jpg | image check }";

int main1(int argc, char* argv[]) {

	// input and check
	CommandLineParser parser(argc, argv, help);
	parser.printMessage();
	if (parser.has("image1") && parser.has("image2")) {
		cout << "has image" << endl;
	}
	else {
		cout << "no image" << endl;
		return -1;
	}

	String input1 = parser.get<String>("image1");
	String input2 = parser.get<String>("image2");
	cout << "Input image1 uri : " << input1 << endl;
	cout << "Input image2 uri : " << input2 << endl;
	if (input1 == "" || input2 == "") {
		cout << "Input image empty. " << endl;
		return -2;
	}

	String chessboard_Photo_Dir1 = samples::findFile(input1 ,false ,true);
	cout << "ChessBoard 1 DIR : " << chessboard_Photo_Dir1 << endl;
	String chessboard_Photo_Dir2 = samples::findFile(input2, false, true);
	cout << "ChessBoard 2 DIR : " << chessboard_Photo_Dir2 << endl;

	if (chessboard_Photo_Dir1 == "" || chessboard_Photo_Dir2 == "") {
		cout << "Input image dir empty. " << endl;
		return -3;
	}
	Mat chessboard_Photo1 = imread(chessboard_Photo_Dir1, IMREAD_UNCHANGED);
	Mat chessboard_Photo2 = imread(chessboard_Photo_Dir2, IMREAD_UNCHANGED);

	cout << "Imagr format: " << chessboard_Photo1.type() << endl;

	// search informations
	Size patternSize(6 , 9);
	Size minSize(2, 2);
	int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK;
	vector<Point2f> chessboard_Corners1;
	vector<Point2f> chessboard_Corners2;

	bool foundCorner1 = findChessboardCorners(chessboard_Photo1 ,patternSize ,chessboard_Corners1 , flags);
	
	if (foundCorner1) {
		cout << "find corner [1]" << endl;
		try{
			cornerSubPix(chessboard_Photo1 , chessboard_Corners1,minSize ,Size(-1 ,-1) , TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
		}
		catch (Exception &ex) {
			return -5;
		}
	}
	else {
		cout << "no find corner [1]" << endl;
		return -1;
	}
	bool foundCorner2 = findChessboardCorners(chessboard_Photo2, patternSize, chessboard_Corners2, flags);
	if (foundCorner2) {
		cout << "find corner [2]" << endl;
		try {
			cornerSubPix(chessboard_Photo2, chessboard_Corners2, minSize, Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.1));
		}
		catch (Exception &ex) {
			return -6;
		}
	}
	else {
		cout << "no find corner [2]" << endl;
		return -2;
	}
	drawChessboardCorners(chessboard_Photo1, patternSize, chessboard_Corners1, foundCorner1);
	drawChessboardCorners(chessboard_Photo2 ,patternSize, chessboard_Corners2, foundCorner2);
	
	// print debug text
	vector<Point2f>::iterator it, end;
	it = chessboard_Corners1.begin();
	end = chessboard_Corners1.end();
	cout << " corner all : "<< (end - it) <<endl;
	for (; it != end ; it ++ ) {
		cout << (*it).x << " " << (*it).y << endl;
	}
	
	// calculate homograpy matrix , method 1
	ClibaHelp* clibHelp = new ClibaHelp;
	vector<Point3f> axis;
	clibHelp->calcChessboards(patternSize ,axis);
	clibHelp->decomposeMatrix(chessboard_Photo1, chessboard_Photo2, patternSize, axis, chessboard_Corners1, chessboard_Corners2);

	// calculate homograpy matrix , method 2
	Mat H = findHomography(chessboard_Corners1 ,chessboard_Corners2);
	cout << "Homography matrix by findHomography():" << H << endl;
	Mat chessboard_Stitch_Photo;
	warpPerspective(chessboard_Photo2 ,chessboard_Stitch_Photo ,H , chessboard_Photo2.size()) ;
	imshow("show chessboard 1", chessboard_Photo2);
	imshow("show chessboard 2" , chessboard_Stitch_Photo);
	waitKey(0);

	delete clibHelp;

	return 0;
}
