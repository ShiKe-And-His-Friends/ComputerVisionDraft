#include <iostream>

#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highui.hpp"

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int main(int argc ,char **argv) {
	std::string in;
	cv::CommandLineParser parser(argc ,argv ,"{@input|../sample/data/corridor.jpg|input image}{help h||show help message}");
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}
	in = parser.get<string>("@input");

	Mat image = imread(in ,IMREAD_GRAYSCALE);
	if (image.empty()) {
		return -1;
	}

	/**
	 * Create FLD detector
	 * Param		Default value	Desription
	 * length_threshold	10		-Segment shorter than this will be discarded
	 * distance_threshold	1.41421356	-A point placed from a hypothesis line
	 * 					segment farther than this will be regarded as an outliner
	 * canny_th1		50		-First threshold for husteresis procedure in Canny()
	 * cann_th2		50		-Second threshold for hysteresis procedure in Canny()
	 * canny_aperture_size	3		-Aperturesize for the sobel operator in Canny()
	 * do_merge		false		-If ture ,incremental merging of segments will be perfomred
	 **/
	int length_threshold = 10;
}
