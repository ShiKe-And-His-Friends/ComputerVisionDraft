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
	cout << "\nControls:\n"
		<< "\tESC - exit\n"
		<< "\tm - change mode GPU <-> CPU\n"
		<< "\tg - convert image to gray or not\n"
		<< "\to - save output image once, or switch on/pff video save\n"
		<< "\t1/q - increase/descrease HOG scale\n"
		<< "\t2/w - increase/decrease levels count\n"
		<< "\t3/e - increase/decrease HOG group threshold\n"
		<< "\t4/r - increase/decrease hit threshold\n"
		<< endl;
		
	make_gray = cmd.has("gray");
	resize_scale = cmd.get<double>("s");
	vdo_source = samples::findFileOrKeep(cmd.get<string>("v"));
	img_source = cmd.get<string>("i");
	output = cmd.get<string>("o");
	camera_id = cmd.get<int>("c");
	
	win_width = 48;
	win_stride_width = 8;
	win_stride_height = 8;
	gr_threshold = 8;
	nlevels = 3;
	hint_threshold = 1.4;
	scale = 1.05;
	gamma_corr = true;
	write_once = false;
	
	cout << "Group threshold: " << gr_threshold << endl;
	cout << "Levels number: " << nlevels << endl;
	cout << "Win width: " << win_width << endl;
	cout << "Win stride: (" << win_stride_width << ", " << win_stride_height << ")" << endl;
	cout << "Hit threshold: " << hint_threshold << endl;
	cout << "Gamma correction: " <<gamma_corr << endl;
	cout << endl;
}

void App::run () {
	running = true;
	VideoWriter video_writer;
	Size win_size(win_width ,win_width * 2);
	Size win_stride(win_stride_width ,win_stride_height);
	
	//Create HOG descriptors and detectors here
	HOGDescriptor hog(win_size ,Size(16 ,16) ,Size(8 ,8) ,Size(8 ,8) ,9 ,1 ,-1
				,HOGDescriptor::L2Hys ,0.2 ,gamma_corr ,cv::HOGDescriptor::DEFAULT_NLEVELS);
	hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
	
	while (running) {
		VideoCapture vc;
		UMat frame;
		if (vdo_source != "") {
			vc.open(vdo_source.c_str());
			if (!vc.isOpened()) {
				throw runtime_error(string("can't open video file: " + vdo_source));
			}
			vc >> frame;
		} else if (camera_id != -1) {
			vc.open(camera_id);
			if (!vc.isOpened()) {
				stringstream msg;
				msg << "can't open camera: " << camera_id ;
				throw runtime_error(msg.str());
			}
			vc >> frame;
		} else {
			imread(img_source).copyTo(frame);
			if (frame.empty()) {
				throw runtime_error(string("can't open image file: " + img_source));
			}
		}
		
		UMat img_aux ,img ,img_to_show;
		//iterate over all frames
		while (running && !frame.empty()) {
			workBegin();
			
			//Change format of the image
			if (make_gray) {
				cvtColor(frame ,img_aux ,COLOR_BGR2GRAY);
			} else {
				frame.copyTo(img_aux);
			}
			
			//Resize image
			if (abs(scale - 1.0) > 0.001) {
				Size sz((int)((double)img_aux.cols / resize_scale) ,(int)((double)img_aux.rows / resize_scale));
				resize(img_aux ,img , sz ,0 ,0 INTER_LINEAR_EXACT);
			} else {
				img = img_aux;
			}
			
			img.coptTo(img_to_show);
			hog.nlevels = nlevels;
			vector<Rect> found;
			
			//Perform HOG classification
			hogWorkBegin();
			hog.detectMultiScale(img ,found ,hint_threshold ,win_stride ,Size(0 ,0) ,scale ,gr_threshold);
			hogWorkEnd();
			
			//Draw positive classification windows
			for (size_t i = 0 ; i < found.size() ; i++ ) {
				rectangle(img_to_show ,found[i] ,Scalar(0 ,255 ,0) ,3);
			}
			putText(img_to_show ,ocl::useOpenCL() ? "Mode OpenCL" : "Mode: CPU" ,Point(5 ,25) ,FONT_HERSHET_SIMPLEX ,1. ,Scalar(255 ,100 ,0) ,2);
			putText(img_to_show ,);
		}
	}
	
}

inline void App::hogWorkBegin() {
	hog_work_begin = getTickCount();
}

inline void App::hogWorkEnd() {
	int64 delta = getTickCount() - hog_work_begin;
	double freq = getTickFrequency();
	hog_work_fps = freq / delta;
}

