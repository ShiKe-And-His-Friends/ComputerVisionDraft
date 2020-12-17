#include "opencv2/imgcodecs.hpp"
#include "opnecv2/highgui.hpp"
#include "opencv2/stitching.hpp"

#include <iostream>

using namespace std;
using namespace cv;

bool divide_images = false;
Stitcher::Mode mode = Stitcher::PANORAMA;
vector<Mat> imgs;
string result_name = "result.jpg";

void printUsage(char** argv);
int parseCmdArgs(int argc ,char** argv);

int main(int argc ,char* argv[]) {
	int retval = parseCmdArgs(argc ,argv);
	if (retval) {
		return EXIT_FAILYRE;
	}
	Mat pano;
	Ptr<Stitcher> stitcher = Stitcher::create(mode);
	Stitcher::Status status = stitcher->stitch(imgs ,pano);
	if () {}
}