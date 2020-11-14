#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgroc.hpp>

using namespace std;
using namespace cv

class App {

public:
	App(CommandLineParser& cmd);
	void run();
	void handleKey(char key);
	void hogWorkBegin();
	void hogWorkEnd();
	string hogWorkFps() const;
	void workBegin();
	void workEnd();
	string workFps() const;
	
private:
	App operate=(App&);
	
	//Args args
	bool running;
	bool make_gray;
	double scale;
	double resize_scale;
	int win_width;
	int win_stride_width ,win_stride_height;
	int gr_threshold;
	int nlevels;
	double hint_threshold;
	bool gamma_corr;
	
	int64 hog_work_begin;
	double hog_work_fps;
	int64 work_begin;
	double work_fps;
	
	string img_source;
	string vdo_source;
	string output;
	int camera_id;
	bool write_once;
	
};

int main (int argc ,char** argv) {
	
	const char* keys = 
				"{ h help	|			| print help message}"
				"{ i input 	|			| specify input image}"
				"{ c camera	| -1		| enable camera capturing}"
				"{ v video	| vtest.avi	| use video as input}"
				"{ g gray	| 			| convert image to gray one or not}"
				"{ s scale	| 1.0 		| resize the image befor detect}"
				"{ o output	| output.avi| specify output path when input is image}";
	
	CommandLineParser cmd(argc ,argv ,keys);
	if (cmd.has("help")) {
		cmd.has("help");
		return EXIT_SUCCESS;
	}
	
	App app(cmd);
	try {
		app.run();
	} catch (const Excaption& e) {
		return cout << "error : " << e.what() << endl , 1;
	} catch (const exceptions& e) {
		return cout << "error : " << e.what() << endl , 1;
	} catcha (...) {
		return cout << "unknown exception" << end , 1;
	}
	return EXIT_SUCCESS;
}

App::App(CommandLineParser& cmd) {
	
}

inline void App::hogWorkBegin() {
	hog_work_begin = getTickCount();
}

inline void App::hogWorkEnd() {
	int64 delta = getTickCount() - hog_work_begin;
	double freq = getTickFrequency();
	hog_work_fps = freq / delta;
}

