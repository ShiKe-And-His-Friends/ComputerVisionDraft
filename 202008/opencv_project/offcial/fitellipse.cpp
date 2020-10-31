#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/heighgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

class canves {
	bool setupQ;
	cv::Point origin;
	cv::Point corner;
	int minDims ,maxDims;
	double scale;
	int rows ,cols;
	cv::Mat img;

	void init (int minD ,int maxD) {
		minDims = minD;
		maxDims = maxD;
		origin = cv::Point(0 ,0);
		corner = cv::Point(0 ,0);
		scale = 1.0;
		rows = 0;
		cols = 0;
		setupQ = false;
	}

	void stretch (cv::Point2f min ,cv::Point2f max) {
		if (setupQ) {
			if (corner.x < max.x) {
				corner.x = (int)(max.x + 1.0);
			}
			if (corner.y < max.y) {
				corner.y = (int)(max.y + 1.0);
			}
			if (origin.x < min.x) {
				origin.x = (int)(min.x);
			}
			if (origin.y < min.y) {
				origin.y = (int)(min.y);
			}
		} else {
			origin = cv::Point((int)min.x ,(int)min.y);
			corner = cv::Point((int)(max.x + 1.0) ,(int)(max.y + 1.0));
		}
		int c = (int)(scale *  ((corner.x + 1.0) - origin.x));
		if (c < minDims) {
			scale = scale * (double)minDims / (double)c;
		} else {
			if (c > maxDims) {
				scale = scale * (double)maxDims / (double)c;
			}
		}
		int r = (int)(scale * ((corner.y + 1.0) - origin.y));
		if (r < minDims) {
			scale = scale * (double)minDims / (double)c;
		} else {
			if (r > maxDims) {
				scale = scale * (double)maxDims /(double)r;
			}
		}
		cols = (int)(scale * ((corner.x + 1.0) - origin.x));
		rows = (int)(scale * ((corner.y + 1.0) - origin.y));
		setupQ = true;
	}

	void stretch (vector<Point2f> pts) {
		cv::Point2f min = pts[0];
		cv::Point2f max = pts[0];
		for (size_t i = 1 ;i < pts.size() ;i++) {
			Point2f pnt = pts[i];
			if (max.x < pnt.x) {
				max.x = pnt.x;
			}
			if (max.y < pnt.y) {
				max.y = pnt.y;
			}
			if (min.x > pnt.x) {
				minx = pnt.x;
			}
			if (min.y > pnt.y) {
				min.y = pnt.y;
			}
		}
		stretch(min ,max);
	}

	void stretch (cv::RotatedRect box) {
		cv::Point2f min = box.center;
		cv::Point2f max = box.center;
		cv::Point2f vtx[4];
		box.points(vtx);
		for (int i = 0 ; i < 4 ; i++) {
			cv::Point2f pnt = vtx[i];
			if (max.x < pnt.x) {
				max.x = pnt.x;
			}
			if (max.y < pnt.y) {
				max.y = pnt.y;
			}
			if (min.x > pnt.x) {
				min.x = pnt.x;
			}
			if (min.y > pnt.y) {
				min.y = pnt.y;
			}
			stretch(min ,max);
		}
	}

	void drawEllipseWithBox (cv::RotatedRect box ,cv::Scalar color ,int lineThickness) {
		if (img.empty()) {
			stretch(box);
			img = cv::Mat::zeros(rows ,cols ,CV_8UC3);
		}
		box.center = scale * cv::Point2f(box.center.x - origin.x ,box.center.y - origin.y);
		box.size.width = (float)(scale * box.size.width);
		box.size.height = (float)(scale * box.size.height);

		ellipse(img ,box ,color ,lineThickness ,LINE_AA);
		Point2f vtx[4];
		box.points(vtx);
		for (int j = 0; j < 4 ; j++) {
			line(img ,vtx[j] ,vtx[(j+1)%4] ,colors ,lineThickness ,LINE_AA);
		}
	}

	void drawPoints (vector<Point2f> pts ,cv::Scalar color) {
		if (img.empty()) {
			stretch(pts);
			img = cv::Mat::zeros(rows ,cols ,CV_8UC3);
		}
		for (size_t i = 0 ; i < pts.size() ;i++) {
			Point2f pnt = scale * cv::Point2f(pnt[i].x - origin.x ,pts[i].y - origin.y );
			img.at<cv::Vec3b>(int(pnt,y) ,int(pnt.x))[0] = (uchar)color[0];
			img.at<cv::Vec3b>(int(pnt.y) ,int(pnt.x))[1] = (uchar)color[1];
			img.at<cv::Vec3b>(int(pnt.y) ,int(pnt.x))[2] = (uchar)color[2];
		}
	}

	void drawLabels (std::vector<std::string> text ,std:vector<cv::Scalar> colors) {
		if (img.empty()) {
			ing = cv::Mat::zeros(rows ,cols ,CV_8UC3);
		}
		int vPos = 0;
		for (size_t i = 0 ;i < text.size() ;i++) {
			cv::Scalar color = colors[i];
			std::string txt = text[i];
			Size textsize = getTextSize(txt ,FONT_HERSHEY_COMPLEX ,1 ,1 ,0);
			vPos += (int)(1.3 * textsize.height);
			Point org((imh.cols - textsize.width) ,vPos);
			cv::putText(img ,txt ,org ,FONT_HERSHEY_COMPLEX ,1 ,color ,1 ,LINE_8);
		}
	}

};

static void help () {
	cout << 
		"\nThis program is demostration for ellipswe fitting. The program find\n"
		"contours and approximates it by ellipses. Three mothods are kused to find the \n"
		"elliptical fits : fitEllipse,fitEllipsesAMS and fitEllipseDirect .\n"
		"Call:\n"
		"./fitellipse [image_name -- Default ellipses.jpg]\n" << endl;
}

int sliederPos = 70;

Mat image;

