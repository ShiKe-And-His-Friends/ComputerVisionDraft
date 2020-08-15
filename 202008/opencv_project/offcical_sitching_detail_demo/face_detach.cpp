#include "pch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

int main(int argc , char* argv[]){

	if (argv[0] == NULL) {
		cout << "input tracker type." << endl;
		return -1;
	}

	string trackerTypes[7] = {"BOOSTING" ,"MIL" ,"KCF" ,"TLD" ,"MEDIANFLOW" ,"MOSSE" ,"CSRT"};
	string trackerType = trackerTypes[5];
	Ptr<Tracker> tracker = TrackerMOSSE::create();

	switch (argv[0]) {
		case '0':
			trackerType = trackerTypes[0];
			tracker = TrackerBoosting::create();
			break;

		case '1':
			trackerType = trackerTypes[1];
			tracker = TrackerMIL::create();
			break;

		case '2':
			trackerType = trackerTypes[2];
			tracker = TrackerKCF::create();
			break;

		case '3':
			trackerType = trackerTypes[3];
			tracker = TrackerTLD::create():
			break;

		case '4':
			trackerType = trackerTypes[4];
			tracker = TrackerMedianFlow::create();
			break;

		case '6':
			trackerType = trackerTypes[6];
			tracker = TrackerCSRT::create();
			break;

		case '5':
		default:
			trackerType = trackerTypes[5];
			tracker = TrackerMOSSE::create();
			break;

	}
	VideoCapture video(0);
	if (!video.isOpened()) {
		cout << "No camera capture." << endl;
		return -1;
	}
	
	Mat frame;
	bool ok = video.read(frame);
	Rect2d bbox(287 ,23 ,86 ,320);
	//bbox = selectROI(frame ,false);  // hand write bound
	rectangle(frame ,bbox ,Scalar(255 ,0 ,0) ,2 ,1);
	imshow("Tracking" ,frame);
	tracker->init(frame ,bbox);
	
	while (video.read(frame)) {
		double timer = (double)getTickCount();
		bool ok = tracker->update(frame ,bbox);
		float fps = getTickFrequency() / ((double)getTickCount() - timer);
		if (ok) {
			rectangle(frame ,bbox ,Scalar(255 ,0 ,0) ,2 ,1);
		} else {
			putText(frame ,"Tracking failure detected" ,Point(100 ,80) ,FONT_HERSHEY_SIMPLEX ,0.75 ,Scalar(0 ,0 ,255) ,2);
		}
		putText(frame ,trackerType + " Tracker" ,Point(100 ,20) ,FONT_HERSHEY_SIMPLEX ,0.75 ,Scalar(50 ,170 ,50) ,2);
		putText(frame ,"FPS:" + to_string(int(fps)) ,Point(100 ,50) ,FONT_HERSHEY_SIMPLEX ,0.75 ,Scalar(50 ,170 ,50) ,2);
		imshow("Tracking" ,frame);
		int k = waitKey(1);
		if (k == 27) {
			break;
		}
	}

	return 0;
}
