#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "CameraCalibrate02.h"

using namespace std;
using namespace cv;


void saveMatInnerData(Mat &gray_Makeself_Photo ,Mat &gray_draw_Photo){
	// file write debug
	FileStorage fileStorage("..//CameraData//20220705_gray_debug.yaml", FileStorage::WRITE);

	int rowIndex = gray_Makeself_Photo.rows;
	int colIndex = gray_Makeself_Photo.cols;
	int channelsNum = gray_Makeself_Photo.channels();

	cout << "gray self make photo channel : " << channelsNum << endl;

	// draw some what
	for (int i = 0; i < rowIndex; i++) {
		for (int j = 0; j < colIndex; j++) {
			if (i % 40 == 0 && j % 40 == 0) {

				fileStorage << "rows " << i << "cols " << j;

				// 1 Byte = 8 bits   1 short = 2 Byte    1 int = 4 Byte
				Vec4w& bgra = gray_Makeself_Photo.at<Vec4w>(i, j);
				fileStorage << "red before " << bgra[0];
				fileStorage << "green before " << bgra[1];
				fileStorage << "blue before " << bgra[2];
				fileStorage << "alpha before " << bgra[3];

				if (channelsNum == 4) {
					bgra[3] = (unsigned short)(0);
				}
				if (channelsNum >= 1 && channelsNum <= 3) {
					unsigned short b = bgra[0];
					unsigned short g = bgra[1];
					unsigned short r = bgra[2];

				}
				// bug: file storage need word instead of number only
				fileStorage << "red after " << bgra[0];
				fileStorage << "green after " << bgra[1];
				fileStorage << "blue after " << bgra[2];
				fileStorage << "alpha after " << bgra[3];
			}
		}
	}

	fileStorage.release();

}

void saveMatData(Mat &gray_Makeself_Photo){
	vector<int> params;
	params.push_back(IMWRITE_JPEG_LUMA_QUALITY);
	params.push_back(90);
	params.push_back(IMWRITE_EXR_COMPRESSION_DWAB);

	bool success = false;
	try {
		success = imwrite("..//CameraData//20220705_gray_makeself.jpg", gray_Makeself_Photo, params);
	}
	catch (const Exception& e) {
		fprintf(stderr, "Exception Write JPEG file error info:", e.what());
	}

	if (success) {
		cout << "save gray self make photo success." << endl;
	}
	else {
		cout << "save gray self make photo failure." << endl;
	}
}

int main(int argc ,char** argv) {

	string lenaPath = samples::findFile("lena.jpg");
	cout << lenaPath << endl;

	Mat lena = imread(lenaPath, IMREAD_COLOR);
	Mat lena_Deepcopy_Photo;
	cout << "rows " << lena.rows << " cols " << lena.cols << endl;
	
	// lena photo deep copy
	lena.copyTo(lena_Deepcopy_Photo);
	lena_Deepcopy_Photo;

	namedWindow("lena");
	imshow("lena", lena_Deepcopy_Photo);
	waitKey(1000);
	destroyAllWindows();

	// format CV_8UC4 CV_32FC4
	Mat gray_Makeself_Photo = Mat(480 ,640 , CV_16UC4,Scalar(64 ,128 ,320 ,255));
	Mat gray_Draw_Photo = Mat(480, 640, CV_16UC4, Scalar(0 ,0 ,0 ,0));
	imshow("gray windows", gray_Makeself_Photo);
	waitKey(1000);

	// save debug data
	saveMatInnerData(gray_Makeself_Photo , gray_Draw_Photo);
	imshow("gray windows", gray_Draw_Photo);
	waitKey(2000);

	// save photo
	saveMatData(gray_Makeself_Photo);

	waitKey(0);

	return 0;
}
