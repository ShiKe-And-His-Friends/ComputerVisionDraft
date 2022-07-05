#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

const char* help = 
	"{ help h | | need help }"
	"{ image1 | left01.jpg | image check }"
	"{ image2 | left04.jpg | image check }";

int main1(int argc, char* argv[]) {

	cv::CommandLineParser parser(argc, argv, help);
	parser.printMessage();
	if (parser.has("image1") && parser.has("image2")) {
		cout << "has image" << endl;
	}
	else {
		cout << "no image" << endl;
		return -1;
	}

	cv::String input1 = parser.get<cv::String>("image1");
	cv::String input2 = parser.get<cv::String>("image2");
	cout << "Input image1 uri : " << input1 << endl;
	cout << "Input image2 uri : " << input2 << endl;
	if (input1 == "" || input2 == "") {
		cout << "Input image empty. " << endl;
		return -2;
	}

	cv::String chessboard_Photo_Dir1 = cv::samples::findFile(input1 ,false ,true);
	cout << "ChessBoard 1 DIR : " << chessboard_Photo_Dir1 << endl;
	cv::String chessboard_Photo_Dir2 = cv::samples::findFile(input2, false, true);
	cout << "ChessBoard 2 DIR : " << chessboard_Photo_Dir2 << endl;

	if (chessboard_Photo_Dir1 == "" || chessboard_Photo_Dir2 == "") {
		cout << "Input image dir empty. " << endl;
		return -3;
	}
	cv::Mat chessboard_Photo1 = cv::imread(chessboard_Photo_Dir1, cv::IMREAD_GRAYSCALE);
	cv::Mat chessboard_Photo2 = cv::imread(chessboard_Photo_Dir2, cv::IMREAD_GRAYSCALE);
	// cv::namedWindow("show chessboard");
	// cv::imshow("show chessboard", chessboard_Photo1);
	// cv::waitKey(1000);
	// cv::imshow("show chessboard", chessboard_Photo2);
	// cv::waitKey(1000);

	cv::Mat homography_Matrix = cv::Mat();

	cv::Mat chessboard_Stitch_Photo;
	cv::warpPerspective(chessboard_Photo2 ,chessboard_Stitch_Photo , homography_Matrix,cv::Size(chessboard_Photo2.cols ,chessboard_Photo2.rows)) ;

	cv::imshow("show chessboard" , chessboard_Stitch_Photo);
	cv::waitKey(0);
	return 0;
}
