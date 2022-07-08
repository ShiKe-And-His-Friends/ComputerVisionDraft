/*
	perspective correction
*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "ClibaHelp.h"

using namespace cv;
using namespace std;

const char* params = 
	"{ help |  | }"
	"{ image1 | left01.jpg | }"
	"{ image2 | left14.jpg | }";



int main(int argc ,char** argv) {

	CommandLineParser  parser(argc ,argv ,params);

	// 读示例图
	string photo_name_1 = parser.get<String>("image1");
	string photo_name_2 = parser.get<String>("image2");
	string photo_dir_1 = samples::findFile(photo_name_1);
	string photo_dir_2 = samples::findFile(photo_name_2);

	Mat photo1 = imread(photo_dir_1, IMREAD_COLOR);
	Mat photo2 = imread(photo_dir_2, IMREAD_COLOR);

	// 配置
	Size patternSize(6, 9);
	Size minSize(2, 2);
	vector<Point3f> aixs;
	ClibaHelp* clibaHelp = new ClibaHelp;
	clibaHelp->calcChessboards(patternSize ,aixs);
	cvtColor(photo1 ,photo1 ,COLOR_BGR2GRAY);
	cvtColor(photo2, photo2, COLOR_BGR2GRAY);

	vector<Point2f> photo_corner_1 , photo_corner_2;
	bool found1 = findChessboardCorners(photo1 ,patternSize , photo_corner_1);
	if (found1) {
		cout << "find corners[1]" << endl;
		cornerSubPix(photo1, photo_corner_1 ,minSize ,Size(-1 ,-1) ,TermCriteria(TermCriteria::COUNT | TermCriteria::EPS ,30 ,0.001));
	}
	else {
		cout << "no find corners[1]" << endl;
		return -1;
	}
	bool found2 = findChessboardCorners(photo2, patternSize, photo_corner_2);
	if (found2) {
		cout << "find corners[2]" << endl;
		cornerSubPix(photo2, photo_corner_2, minSize, Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.001));
	}
	else {
		cout << "no find corners[2]" << endl;
		return -2;
	}

	// 计算单应矩阵
	Mat H = findHomography(photo_corner_1 ,photo_corner_2);
	cout << "H :" << endl << H << endl;

	// 拼图看效果
	Mat photo_warp;
	warpPerspective(photo1 ,photo_warp ,H ,photo1.size());
	Mat photo_draw_warp;
	hconcat(photo2 ,photo_warp ,photo_draw_warp);
	imshow("warp draw" ,photo_draw_warp);
	waitKey(1000);
	destroyAllWindows();

	// 连线
	RNG rng(0xffffffff);
	for (size_t i = 0; i < photo_corner_1.size(); i++) {
		Mat point1 = (Mat_<double>(3 ,1) << photo_corner_1[i].x , photo_corner_1[i].y ,1);
		// Mat point2 = (Mat_<double>(3 ,1) << photo_corner_2[i].x ,photo_corner_2[i].y ,1);
		Mat point2 = H * point1;
		point2 /= point2.at<double>(2);

		Point end_point((int)(photo1.cols + point2.at<double>(0)), (int)(point2.at<double>(1)));
		line(photo_draw_warp , photo_corner_1[i] , end_point , clibaHelp->randomColor(rng) ,3);
	}
	imshow("match draw", photo_draw_warp);
	waitKey(1000);
	destroyAllWindows();

	// 释放
	delete clibaHelp;

	return 0;
}