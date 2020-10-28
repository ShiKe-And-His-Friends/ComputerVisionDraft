#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help () {
	cout << "\nThis program demonstrated the floodFill() function.\n"
		"Call:\n"
		"./filldemo [image_name -- Default fruits.jpg] \n" << endl;

	cout << "Hot keys:\n"
		"\tEsc - quit the program\n"
		"\tc - switch color/graysscale mode\n"
		"\tm - switch mask mode\n"
		"\tr - restore the original image\n"
		"\ts - use null-range floodfill\n"
		"\tf - use gradient floodfill with fixed(absolute) range\n"
		"\tg - use gradient floodfill with floating(relative) range\n"
		"\t4 - use 4-connectivity mode\n"
		"\t8 - use 8-connectivety mode\n" << endl;
}

Mat image0 ,image ,gray ,mask;
int fillMode = 1;
int loDiff = 20 ,upDiff = 20;
int connectivty = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;

static void onMouse (int event ,int x ,int y ,int ,void*) {
	if (event != EVENT_LBUTTONDOWN) {
		return ;
	}
	Point seed = Point(x ,y);
	int lo = fillMode == 0 ? 0 : loDiff ;
	int up = fillMode == 0 ? 0 : upDiff ;
	int flags = connectivty + (newMaskVal << 8) + (fillMode == 1 ? FLOODFILL_FIXED_RANGE : 0);
	int b = (unsigned)theRNG() & 255;
	int g = (unsigned)theRNG() & 255;
	int r = (unsigned)theRNG() & 255;
	Rect ccomp;
	Scalar newVal = isColor ? Scalar(b ,g ,r) : Scalar(r*0.299 + g*0.587 + b*0.114);
	Mat dist = isColor ? image : gray;
	int area;
	if (useMask) {
		threshold(mask ,mask ,1 ,128 ,THRESH_BINARY);
		area = floodFill(dist ,mask ,seed ,newVal ,&ccomp ,Scalar(lo ,lo ,lo) ,Scalar(up ,up ,up) ,flags);
		imshow("mask" ,mask);
	} else {
		area = floodFill(dist ,seed ,newVal ,&ccomp ,Scalar(lo ,lo ,lo) ,Scalar(up ,up ,up) ,flags);
	}
	imshow("image" ,dist);
	cout << area << " pixel were repainted\n";
}

int main (int argc ,char** argv) {
	cv::CommandLineParser parser (argc ,argv ,"{help h || show help message}{@image|fruits.jpg | input image}");
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}
	string filename = parser.get<string>("@image");
	image0 = imread(samples::findFile(filename) ,1);
	if (image0.empty()) {
		cout << "Image empty\n";
		parser.printMessage();
		return 0;
	}
	help();
	image0.copyTo(image);
	cvtColor(image0 ,gray ,image0.cols + 2 ,CV_8UC1);
	namedWindow("image" ,0);
	createTrackbar("lo_diff" ,"image" ,&loDiff ,255 ,0);
	createTrackbar("up_diff" ,"image" ,&upDiff ,255 ,0);

	setMouseCallback("image" ,onMouse ,0);

	for (;;) {
		imshow("image" ,isColor ? image : gray);
		char c = (char)waitKey(0);
		if (c == 27) {
			cout << "Exit ...\n";
			break;
		}
		switch (c) {
			case 'c':
				if (isColor) {
					cout << "Grayscale mode is set\n";
					cvtColor(image0 ,gray ,COLOR_BGR2GRAY);
					mask = Scalar::all(0);
					isColor = false;
				} else {
					cout << "Color mode us set.\n";
					image0.copyTo(image);
					mask = Scalar::all(0);
					isColor = true;
				}
				break;

			case 'm':
				if (useMask) {
					destroyWindow("mask");
					useMask =false;
				} else {
					namedWindow("mask" ,0);
					mask = Scalar::all(0);
					imshow("mask" ,mask);
					useMask = true;
				}
				break;

			case 'r':
				cout << "Original image is restored\n";
				image0.copyTo(image);
				cvtColor(image ,gray ,COLOR_BGR2GRAY);
				mask = Scalar::all(0);
				break;

			case 's':
				cout << "Simple floodfill mode is set\n";
				fillMode = 0;
				break;

			case 'f':
				cout << "fixed Range floodfill mode is set\n";
				fillMode = 1;
				break;

			case 'g':
				cout << "Gradlient (floating range) floodfill mode is set\n";
				fillMode = 2;
				break;

			case '4':
				cout << "4*connectivity mode is set\n";
				connectivity = 4;
				break;

			case '8':
				cout << "8*connectivity mode is set\n";
				connectivity = 8;
				break;
		}
	}

	return 0;
}

