#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const char* help = 
	"{ help h | | need help }"
	"{ image1 | left01.jpg | image check }"
	"{ image2 | left04.jpg | image check }";

int main(int argc, char* argv[]) {

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
	Mat chessboard_Photo1 = imread(chessboard_Photo_Dir1, IMREAD_GRAYSCALE);
	Mat chessboard_Photo2 = imread(chessboard_Photo_Dir2, IMREAD_GRAYSCALE);


	Mat homography_Matrix = Mat();

	Mat chessboard_Stitch_Photo;
	//warpPerspective(chessboard_Photo2 ,chessboard_Stitch_Photo , homography_Matrix,Size(chessboard_Photo2.cols ,chessboard_Photo2.rows)) ;

	//imshow("show chessboard" , chessboard_Stitch_Photo);
	waitKey(0);

	return 0;
}
