#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

// 跳步打印图片到YAML文件，同时用坐标画另一新图片
void saveMatInnerData(Mat &gray_Makeself_Photo,Mat &gray_draw_Photo){
	// file write debug
	FileStorage fileStorage("..//CameraData//20220705_gray_debug.yaml", FileStorage::WRITE);

	int rowIndex = gray_Makeself_Photo.rows;
	int colIndex = gray_Makeself_Photo.cols;
	int channelsNum = gray_Makeself_Photo.channels();

	cout << "gray self make photo channel : " << channelsNum << endl;

	// draw some what
	for (int i = 0; i < rowIndex; i++) {
		for (int j = 0; j < colIndex; j++) {
			// 1 Byte = 8 bits   1 short = 2 Byte    1 int = 4 Byte
			Vec4w& bgra_Gray_Makeself = gray_Makeself_Photo.at<Vec4w>(i, j);
			Vec4w& bgra_Gray_Draw_Photo = gray_draw_Photo.at<Vec4w>(i, j);
			if (channelsNum == 4) {
				bgra_Gray_Makeself[3] = (unsigned short)(0);
			}
			if (channelsNum >= 1 && channelsNum <= 4) {
				unsigned short b = bgra_Gray_Makeself[0];
				unsigned short g = bgra_Gray_Makeself[1];
				unsigned short r = bgra_Gray_Makeself[2];
				bgra_Gray_Draw_Photo[0] = saturate_cast<unsigned short>(((float)(colIndex - j) / (float)(colIndex)) * USHRT_MAX); // blue
				bgra_Gray_Draw_Photo[1] = saturate_cast<unsigned short>(((float)(colIndex - j) / (float)(colIndex)) * USHRT_MAX); // green
				bgra_Gray_Draw_Photo[2] = saturate_cast<unsigned short>(((float)(rowIndex - i) / (float)(rowIndex)) * USHRT_MAX); // red
				bgra_Gray_Draw_Photo[3] = saturate_cast<unsigned short>(0.8 * (g + r)); //alpha
			}

			if (i % 40 == 0 && j % 40 == 0) {
				fileStorage << "rows " << i << "cols " << j;
				fileStorage << "red before " << bgra_Gray_Makeself[0];
				fileStorage << "green before " << bgra_Gray_Makeself[1];
				fileStorage << "blue before " << bgra_Gray_Makeself[2];
				fileStorage << "alpha before " << bgra_Gray_Makeself[3];

				// bug: file storage need word instead of number only
				fileStorage << "red after " << bgra_Gray_Draw_Photo[0];
				fileStorage << "green after " << bgra_Gray_Draw_Photo[1];
				fileStorage << "blue after " << bgra_Gray_Draw_Photo[2];
				fileStorage << "alpha after " << bgra_Gray_Draw_Photo[3];
			}
		}
	}
	fileStorage.release();

}

// 二值化图片
void binarilation(Mat& mat) {

}

// 保存图片到本地
void saveMatData(Mat &lena_Deepcopy_Photo, Mat &gray_Makeself_Photo, Mat &gray_Draw_Photo){
	vector<int> params;
	params.push_back(IMWRITE_JPEG_QUALITY);
	params.push_back(90);
	params.push_back(IMWRITE_EXR_COMPRESSION_DWAB);

	bool success = false;
	try {
		success = imwrite("..//CameraData//20220705_gray_makeself.jpg", gray_Makeself_Photo, params);
	}
	catch (const Exception& e) {
		fprintf(stderr, "Exception Write JPEG file [1] error info:", e.what());
	}

	try {
		success = imwrite("..//CameraData//20220705_gray_draw.jpg", gray_Draw_Photo, params);
	}
	catch (const Exception& e) {
		fprintf(stderr, "Exception Write JPEG file [2] error info:", e.what());
	}

	try {
		success = imwrite("..//CameraData//20220705_lena_deep_copy.jpg", lena_Deepcopy_Photo, params);
	}
	catch (const Exception& e) {
		fprintf(stderr, "Exception Write JPEG file [3] error info:", e.what());
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

	namedWindow("lena");
	imshow("lena", lena_Deepcopy_Photo);
	waitKey(1000);
	destroyAllWindows();

	// format CV_8UC4 CV_32FC4
	Mat gray_Makeself_Photo = Mat(480 ,640 , CV_16UC4,Scalar(64 ,128 ,320 ,255));
	Mat gray_Draw_Photo = Mat(480, 640, CV_16UC4, Scalar(0 ,0 ,128 ,0));
	imshow("gray windows", gray_Makeself_Photo);
	waitKey(1000);
	destroyAllWindows();

	// swap
	binarilation(lena_Deepcopy_Photo);

	// save data
	saveMatInnerData(gray_Makeself_Photo , gray_Draw_Photo);
	imshow("gray windows", gray_Draw_Photo);
	waitKey(2000);
	destroyAllWindows();

	// save photo
	saveMatData(lena_Deepcopy_Photo ,gray_Makeself_Photo , gray_Draw_Photo);

	waitKey(0);

	return 0;
}
