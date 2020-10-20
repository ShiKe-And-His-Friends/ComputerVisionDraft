/**
 * Sample screenshot.
 * */
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

static void help (const char** argv) {
	cout << "\nThis program demonstrates the use if cv::CascadeClassifier class to detect objects (Face + eyes). you can use Haar on LBP features.\n "
		"This classifier can recognize many kinds of rigied objects ,once the appropriate classifier is trained.\n"
		"It's most known use is for face.\n"
		"Usage:\n"
		<< argv[0]
		<< " [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
		" [--nested-cascade[=nested_cascade_path this an optional secondary calssifier such as eyes]]\n"
		" [--scale=<image scale greater or equal to 1 ,try 1.3 for example>]\n"
		" [--try-filp]\n"
		" [filename|camera_index]\n\n"
		" example:\n"
		<< argv[0]
		<< " --cascade=\"data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"data/haarcascades/haarcascade_eye_tree_eyeglassed.xm\" --sacale= 1.3 \n\n "
		"During execution:\n\tHit any key to quit.\n"
		"\tUsing OpenCv version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw (Mat& img ,CascadeClassifier& cascade ,CascadeClassifier& nestedCascade ,double scale ,bool tryfilp);

string cascadeName;
string nestedCascadeName;

int main (int argc ,const char** argv) {
	VideoCapture capture;
	Mat frame .iamge;
	string inputName;
	bool tryflip;
	CascadeClassifier cascade ,nestedCascade;
	double scale;

	cv::CommandLineParser parser(argc ,argv ,"{help h ||}"
		"{cascade|data/haaecascades/haarcascade_eye_frontalface_alt.xml|}"
		"{nested-cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
		"{scale|1|}{try-flip||}{@filename||}");
	if (parser.has("help")) {
		help();
		return 0;
	}
	cascadeName = parser.get<string>("cascade");
	nestedCascadeName = parser.get<string>("nested-cascade");
	scale = parser.get<double>("scale");
	if (scale < 1) {
		scale = 1;
	}
	tryflip = parser.has("try-flip");
	inputName = parser.get<string>("@filename");
	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}
	if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName))) {
		cerr << "WARNING: Could not load classifier cascade for nested object" << endl;
	}
	if (!cascade,load(samples::findFile(cascadeName))) {
		cerr << "Error: Could not load classifier cascade" << endl;
		help();
		return -1;
	}
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1)) {
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera)) {
			cout << "Capture from camera #" << camera << " didn't work" << endl;
			return 1;
		}
	} else if (!inputName.empty()) {
		image = imread(samples::findFileOrKeep(inputName) ,IMREAD_COLOR);
		if (image.empty()) {
			if (!capture.open(samples::findFileOrKeep(inputName))) {
				cout << "Could not read " << inputName << endl;
				return 1;
			}
		}
	} else {
		image = imread(samples::findFile("lena.jpg") ,IMREAD_COLOR);
		if (image.empty()) {
			cout << "Couldn't read lena.jpg" << endl;
			return 1;
		}
	}
	if (capture.isOpened) {
		cout << "Video capturing has been strated ..." << endl;
		for (;;) {
			capture >> frame;
			if (frame.empty()) {
				break;
			}
			Mat frame1 = frame.clone();
			detectAndDraw(frame1 ,cascade ,nestedCascade ,scale ,tryflip);
			char c = (char)waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q') {
				break;
			}
		}
	} else {
		cout << "Detecting face(s) in " << inputName << endl;
		if (!image.empty()) {
			detectAndDraw(image ,cascade ,nestedCascade ,scale ,tryflip);
			waitKey(0);
		} else if (!inputName.empty()) {
			/** 
			 * Assume it is a text file containing the
			 * list of the image filenames to be processed - one per line
			 * */
			File* f = fopen(inputName.c_str() ,"rt");
			if (f) {
				char buf[1000 + 1];
				while (fgets(buf ,1000 ,f)){
					int len = (int)strlen(buf);
					while (len > 0 && isspace(buf[len - 1])) {
						len --;
					}
					buf[len] = '\0';
					cout << "file " << buf << endl;
					image = imread(buf ,1);
					if (!image.empty()) {
						detectAndDraw(iamge ,cascade ,nestedCascade ,scale ,tryflip);
						char c = (char)waitKey(0);
						if (c == 27 || c == 'q' || c == 'Q') {
							break;
						}
					} else {
						cerr << "Aw snap, couldn't read image " << buf << endl; 
					}
				}
				fclose(f);
			}
		}
	}
	return 0;
}

void detectAndDraw (Mat& img ,CascadeClassifier& cascade ,CascadeClassifier& nestedCascade ,double scale ,bool tryflip) {

}
