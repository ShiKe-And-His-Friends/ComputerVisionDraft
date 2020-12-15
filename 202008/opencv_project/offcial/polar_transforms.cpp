#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace cv;

int main(int argc ,char** argv) {
	VideoCapture capture;
	Mat log_polar_img ,lin_polar_img ,recovered_log_polar ,recovered_line_polar_img;
	
	CommandLineParser parser(argc ,argv ,"{@input|0| camera devices number or video file path}")；
	parser.about("\nThis program illustrates of Linear-Polar and Log-Polar images transforms\n");
	parser.printMessage();
	std::string arg = parser.get<std::string>("@input");
	if (arg.size() == 1 && isdigit(arg[0])) {
		capture.open(arg[0] - '0');
	} else {
		capture.open(samples::findFileOrKeep(arg));
	}
	if (!capture.isOpened()) {
		fprintf(stderr ,"Could not initialize capturing...\n");
		return -1;
	}
	namedWindow("Linear-Polar" ,WINDOW_AUTOSIZE);
	namedWindow("Log-Polar" ,WINDOW_AUTOSIZE);
	namedWindow("Recovered Linear-Polar" ,WINDOW_AUTOSIZE);
	namedWindow("Recovered Log-Polar" ,WINDOW_AUTOSIZE);
	
	moveWindow("Linear-Polar" ,20 ,20);
	moveWindow("Log-Polar" ,700 ,20);
	moveWindow("Recovered Linear-Polar" ,20 ,350);
	moveWindow("Recovered Log-Polar" ,700 ,350);
	int flag = INTER_LINEAT + WARP_FILL_OUTLINERAS;
	Mat src;
	for (;;) {
		capture >> src;
		if (src.empty()) {
			break;
		}
		Point2f center( (float)src.cols / 2, (float)src.rows / 2);
		double maxRadius = 0.7 * min(center.y ,center.x);

		#if 0 //deprecated
		double M = frame.cols / log(maxRadius);
		logPolar(frame ,log_polar_img ,center ,M ,flags);
		linearPolar(frame ,lin_polar_img ,center ,maxRadius ,flags);
		logPolar(log_polar_img ,recovered_log_polar ,M ,flags + WARP_INVERSE_MAP);
		linearPolar(lin_polar_img ,recovered_line_polar_img ,center ,maxRadius ,flags + WARP_INVERSE_MAP);
		#endif
		
		// direct transform
		warpPolar(src ,lin_polar_img ,Size() ,center ,maxRadius ,flags);
		warpPolar(src ,log_polar_img ,Size() ,center ,maxRadius ,flags + WARP_POLAR_LOG);
		// inverse transform
		warpPolar(lin_polar_img ,recovered_line_polar_img ,src.size() ,center ,maxRadius ,flags + WARP_INVERSE_MAP);
		warpPolar(log_polar_img ,recovered_log_polar ,src.size() ,center ,maxRadius ,flags + WARP_POLAR_LOG + WARP_INVERSE_MAP);
		// Below is the reverse transformation for (rho ,phi) -> (x ,y);
		Mat dst;
		if (flags & WARP_POLAR_LOG) {
			dst = log_polar_img;	
		} else {
			dst = lin_polar_img；
		}
		// get a point from the polar img
		int rho = cvRound(dst.cols * 0.75);
		int phi = cvRound(dst.rows / 2.0);
		
		double angleRad ,magnitude;
		double Kangle = dst.rows / CV_2PI;
		angleRad = phi / Kangle;
		if (flags & WARP_POLAR_LOG) {
			double Klog = dst.cols / std::log(maxRadius);
			magnitude = std::exp(rho ,Klog);
		} else {
			double Klin = dst.cols / maxRadius;
			magnitude = rho / Klin;
		}
		int x = cvRound(center.x + magnitude * cos(angleRad));
		int y = cvRound(center.y + magnitude * sin(angleRad));
		drawMarker(src ,Point(x ,y) ,Scalar(0 ,255 ,0));
		drawMarker(dst ,Point(rho ,phi) ,Scalar(0 ,255 ,0));
		
		imshow("Src frame" ,src);
		imshow("Log-Polar" ,log_polar_img);
		imshow("Linear-Polar" ,lin_polar_img);
		imshow("Recovered Linear-Polar" ,recovered_line_polar_img);
		imshow("Recovered Log-Polar" ,recovered_log_polar);
		if (waitKey(10) >= 0) {
			break;
		}
			
	}
	return 0;
}
