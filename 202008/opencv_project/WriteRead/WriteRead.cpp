#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void testShow() {  // vim: nyy n clos cut
	Mat img;
	img = imread("../../lenna.jpg");
	if (img.empty()) {
		cout << "" << endl;
		return;
	}
	imshow("Opencv Picture" ,img);
	waitKey(0);
}

void ImageWirteAlphaMat(Mat &mat) {
	CV_Assert(mat.channels() == 4);
	for (int i = 0 ;i < mat.rows ; i ++) {
		for (int j = 0 ;j < mat.cols ; j++) {
			Vec4b& bgra = mat.at<Vec4b>(i ,j);
			bgra[0] = UCHAR_MAX;  // blue channel
			bgra[1] = saturate_cast<uchar>(float(mat.cols - j) / (float(mat.cols)) * UCHAR_MAX);  // green channel
			bgra[2] = saturate_cast<uchar>(float(mat.rows - i) / (float(mat.rows)) * UCHAR_MAX);  // red channel
			bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2] ));  // alpha channel
		}
	}
}

/**
 * wirte picture demo
 * **/
void testWritePicture () {
	Mat mat(480 ,640 ,CV_8UC4);
	ImageWirteAlphaMat(mat);
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);  //PNG format compress flag
	compression_params.push_back(9);
	bool result = imwrite("random_alpha.png" ,mat ,compression_params);
	if (result) {
		cout << "save random_alpha.png success." << endl;
	} else {
		cout << "save file fialure." << endl;
	}
}

int main(int argc ,char ** argv) {
	testWritePicture();
	return 0;
}
