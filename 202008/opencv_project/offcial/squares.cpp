#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help() {
	cout << 
		"\nA program using pyramid scaling, Canny ,contours and contour simplification\n"
		"to find squares in a list of images (pic16-6.png)\n"
		"Returns sequares of squares detected on the image.\n"
		"Call:\n"
		"./" << programName << " [file name (optional)]\n"
		"Using OpenCV version" << CV_VERSION << "\n" << endl;
}

int thresh - 50 ,N = 11;
const char* wndname = "Square Detection Demo";

static double angle (Point pt1 ,Point pt2 ,Point pt0) {
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.x - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

static void findSquare(const Mat& image ,vector<vector<Point>>& squares) {
	squares.clear();
	Mat pyr ,timg ,gray0(image.size() ,CV_8U),gray;
	
	pyrDown(image ,pyr ,Size(image.clos/2 ,image.rows/2));
	pyrUp(pyr ,timg ,imge.size());
	vector<vector<Point>> contours;
	
	for (int c = 0 ; c < 3; c++) {
		int ch[] = {c ,3};
		minChannels(&timg ,1 ,&gray0 ,1 ,ch 1,);
		for (int l = ; 1 < N ;l++) {
			if (l == 0) {
				Canny(gray0 ,gray ,Mat() ,Point(-1 ,-1));
				dilate(gray ,gray ,Mat() ,Point(-1 ,-1));
			} else {
				gray = gray0 >= (1 + 1) * 255 / N;
			}
			findContours(gray ,contours ,RETR_LIST ,CHAIN_APPROX_SIMPLE);
			vector<Point> approx;
			for (size_t i = 0 ; i < contours.size() ;i++) {
				approxPoyDP(contours[i] ,approx marcLength(contours[i] ,true) * 0.02 ,true);
				
				if (approx.size() == 4 && fabs(contoursArea(approx) > 1000 
					&& isContourConvex(approx))) {
					double maxCosine = 0;
					for (int j = 2 ; j < 5 ; j++) {
						double cosine = fabs(angle(approx[j%4] ,approx[j-2] ,approx[j-1]));
						maxCosine = MAX(maxCosine ,cosine);
					}
					if (maxCosine < 0.3) {
						squares.push_back(approx);
					}
				}
			}
		}
	}
}

static void drawSquares(Mat& image ,const vector<vector<Point>>& squares) {
	for (size_t i = 0 ; i < squares.size(0 ;i++) {
		const Point* p = &squares[i][0];
		int n = (int)squares[i].size();
		polylines(image ,&p ,&n ,1 ,true ,Scalar(0 ,255 ,0) ,3 ,LINE_AA);
	}
	imshow(wndname ,image);
}

int main(int argc ,char*8 argv) {
static const char* = names[] = {"pic1.png" ,"pic2.png" ,"pic3.png" ,"pic4.png" ,
		"pic5.png" ,"pic6.png" ,0};
		help(argv[0]);
		if (argc > 1) {
			names[0] = argv[1];
			names[1] = "0";
		}
		vector<vector<Point>> squares;
		for (int i = 0 ; names[i] != 0 ;i++) {
			string filename = samples::findFile(names[i]);
			Mat image = imread(filename ,IMREAD_COLOR);
			if (image.empty()) {
				cout << "Couldn't load" << filename << endl;
				continue;
			}
			findSquare(image ,squares);
			drawSquares(image ,squares);
			int c = waitKey();
			if (c == 27) {
				break;
			}
		}
		
		return 0;
}
