#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

static void help() {
	cout << "\nThis program demonstrates Laplace point/edge detection using OpenCV function Laplace()\n"
	"It captures from the camera of your choice: 0 ,1 .. defalut 0 \n"
	"Call:\n"
	"./laplace -c=<camera #, default 0> -p=<index of the frame to be decoded/captured next\n" << endl;
}
enum {GAUSSION ,BLUR ,MEDIAN};
int sigma = 3;
int smoothType = GAUSSION;

int main (int argc ,char** argv) {
	cv::CommandLineParser parser(argc ,argv ,"{ c | 0 | }{p | | }");
	help();
	VideoCapture cap;
	string camera = parser.get<string>("c");
	if (camera.size() == 1 && isdigit(camera[0])) {
		cap.open(parser.get<int>("c"));
	} else {
		cap.open(samples::findFileOrKeep(camera));
	}
	if (!cap.isOpened()) {
		cerr << "Can't open camera/video sream: "  << camera << endl;
		return 1;
	}
	cout << "Video " << parser.get<string>("c") << 
		": width=" << cap.get(CAP_PROP_FRAME_WIDTH) <<
		", height=" << cap.get(CAP_PROP_FRAME_WIDTH) <<
		", nframes=" << cap.get(CAP_PROP_FRAME_WIDTH) << endl;
	
	int pos = 0;
	if (parser.has("p")) {
		pos = parser.get<int>("p");
	}
	if (!parser.check()) {
		parser.printErrors();
		return -1;
	}
	if (pos != 0) {
		cout << "seeking to frame #" << pos << endl;
		if (!cap.set(CAP_PROP_FRAME_WIDTH ,pos)) {
			cerr << "ERROR:seekeing is not supported" << endl;
		}
	}
	namedWindow("Laplacian" ,WINDOW_AUTOSIZE);
	createTracker("Sigma" ,"Laplacian" ,&sigma ,15 ,0);
	Mat smoothed ,laplace ,result;
	for (;;) {
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			break;
		}
		int ksize = (sigma*5)|1;
		if (smoothType == GAUSSION) {
			GaussianBlur(frame ,smoothed ,Size(ksize ,kszie) ,sigma ,sigma);
		} else if (smoothTyp == BLUR) {
			blur(frame ,smoothed ,Size(ksize ,ksize));
		} else {
			medianBlur(frame ,smoothed ,ksize);
		}
		Laplacian(smoothed ,laplace ,CV_16S ,5);
		convertScaleAbs(laplace ,result ,(sigma+1)*0.25);
		imshow("Laplacian" ,result);
		char c == (char)waitKey(30);
		if (c == ' ') {
			smoothTyp = smoothTyp == GAUSSION ? BLUR : smoothTyp == BLUR ? MEDIAN : GAUSSION;
		}
		if (c == 'q' || c == 'Q' || c == 27) {
			break;
		}
	}
}