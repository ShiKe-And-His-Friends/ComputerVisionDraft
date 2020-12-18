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
		return EXIT_FAILURE;
	}
	Mat pano;
	Ptr<Stitcher> stitcher = Stitcher::create(mode);
	Stitcher::Status status = stitcher->stitch(imgs ,pano);
	if (status != Stitcher::OK) {
		cout << "Can't sitich image ,error code = " << int(status) << endl;
		return EXIT_FAILURE;
	}
	imwrite(result_name ,pano);
	cout << "stitching complted successfully\n" << result_name << " saved!";
	return EXIT_SUCCESS;
}

void printUsage(char** argv) {
	cout << "Images stitvher.\n\n" << "Usage:\n" << argv[0] << " [Flags] img1 img2 [...imgN]\n\n"
		"Flags:\n"
		"	--d3\n"
		"	internally creates three chunks of each image to increase stitching success\n"
		"	--mode (panorama|scans)"
		"	Determines configuration of stitcher.The default is 'panorama',\n"
		"	modesuitable for creating photo panpramas.Option'scans' is suitable\n"
		"	for stitching materials under affine transformation,such as scans.\n"
		"	--output <result_img>\n"
		"	The default is 'result.jpg'\n\n"
		"Example usage :\n" << argv[0] << "	--d3 --try_use_gpu yes --mode scans img1.jpg img2.jpg\n\n"
}

int parseArgs(int argc ,char** argv) {
	if (argc == 1) {
		printUsage(argv);
		return EXIT_FAILURE;
	}
	for (int i = 1 ; i < argc ; i++) {
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?") {
			printUsage(argv);
			return EXIT_FAILURE;
		} else if (string(argv[i]) == "--d3") {
			divide_images = true;
		} else if (string(argv[i]) == "--output") {
			result_name = argv[i+1];
			i++;
		} else if (string(argv[i]) == "--mode") {
			if (string(argv[i+1] == "panorama")) {
				mode = Stitcher::PANORAMA;
			} else if (string(argv[i+1] == "scans")) {
				mode = Stitcher::SCANS;
			} else {
				cout << "Bad --mode flag value\n";
				return EXIT_FAILURE;
			}
			i++;
		} else {
			Mat img = imread(samples::findFile(argv[i]));
			if (img.empty()) {
				cout << "Cann't read image '" << argv[i] << "'\n";
				return EXIT_FAILURE;
			}
			if (divide_images) {
				Rect rect(0 ,0 ,img.cols /2 ,img.rows);
				imgs.push_back(img(rect).clone());
				rect.x = img.cols / 3;
				imgs.push_back(img(rect).clone());
				rect.x = img.cols / 2;
				imgs.push_back(img(rect).clone());
			} else {
				imgs.push_back(img);
			}
		}
	}
	return EXIT_SUCCESS;
}