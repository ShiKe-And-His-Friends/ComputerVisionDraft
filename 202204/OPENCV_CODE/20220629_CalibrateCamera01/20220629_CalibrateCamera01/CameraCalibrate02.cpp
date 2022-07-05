#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc ,char** argv) {

	string lenaPath = samples::findFile("lena.jpg");
	cout << lenaPath << endl;

	Mat lena = imread(lenaPath, IMREAD_COLOR);
	Mat lena_Deepcopy_Photo;
	cout << "rows " << lena.rows << " cols " << lena.cols << endl;
	
	lena.copyTo(lena_Deepcopy_Photo);
	lena_Deepcopy_Photo;

	namedWindow("lena");
	imshow("lena", lena_Deepcopy_Photo);
	waitKey(1000);
	destroyAllWindows();

	// format CV_8UC4 CV_32FC4
	Mat gray_Makeself_Photo = Mat(80 ,60 , CV_16UC4,Scalar(64 ,128 ,320 ,255));
	imshow("gray windows", gray_Makeself_Photo);
	waitKey(1000);
	int rowIndex = gray_Makeself_Photo.rows;
	int colIndex = gray_Makeself_Photo.cols;
	int channelsNum = gray_Makeself_Photo.channels();

	cout << "gray self make photo channel : " << channelsNum << endl;
	// draw some what
	for (int i = 0; i < rowIndex; i++ ) {
		for (int j = 0; j < colIndex; j ++) {
			Vec4b& bgra = gray_Makeself_Photo.at<Vec4b>(i, j);
			if (channelsNum == 4) {
				// bgra[3] = 0;
			}
			if (channelsNum >= 1 && channelsNum <= 3) {
				int b = bgra[0];
				int g = bgra[1];
				int r = bgra[2];

				

				//bgra[0] = UCHAR_MAX; // blue light
			}
		}
	}

	vector<int> params;
	params.push_back(IMWRITE_JPEG_LUMA_QUALITY);
	params.push_back(90);
	params.push_back(IMWRITE_EXR_COMPRESSION_DWAB);

	bool success = imwrite("..//CameraData//20220705_gray_makeself.jpg" , gray_Makeself_Photo ,params);
	if (success) {
		cout << "save gray self make photo success." << endl;
	}
	else {
		cout << "save gray self make photo failure." << endl;
	}

	waitKey(0);

	return 0;
}
