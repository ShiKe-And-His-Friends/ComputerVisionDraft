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

