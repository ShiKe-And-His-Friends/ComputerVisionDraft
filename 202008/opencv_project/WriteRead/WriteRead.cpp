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

/**
 * video capture
 * **/
void videoCapture () {
	system("color F0");  // control GUI backgound colors
	VideoCapture video("/home/shike/Videos/sugar.mp4");
	if (video.isOpened()) {
		cout << "width = " <<  video.get(CAP_PROP_FRAME_WIDTH) << "\n"
		<< "height = " << video.get(CAP_PROP_FRAME_HEIGHT) << "\n"
		<< "frame = " << video.get(CAP_PROP_FRAME_COUNT) << endl;
	} else {
		cout << "video capture failure." << endl;
	}
	while (1) {
		Mat frame;
		video >> frame ;
		if (frame.empty()) {
			break;
		}
		imshow("OpenCV Video" ,frame);
		waitKey(1000 / video.get(CAP_PROP_FPS));
	}
	waitKey();
}

/**
 * video write
 * **/
void videoWrite () {
	Mat img;
	VideoCapture video(0);  // Linux camera
	if (video.isOpened()) {
		cout << "width = " <<  video.get(CAP_PROP_FRAME_WIDTH) << "\n"
		<< "height = " << video.get(CAP_PROP_FRAME_HEIGHT) << "\n"
		<< "frame = " << video.get(CAP_PROP_FRAME_COUNT) << endl;
	} else {
		cout << "video capture failure." << endl;
	}
	video >> img;
	if (img.empty()) {
		cout << "video image empty." << endl;
		return;
	}
	bool isColorChannel = (img.type() == CV_8UC3);
	VideoWriter write;
	int codec = VideoWriter::fourcc('M' ,'J' ,'P' ,'G'); // choose decode format
	double fps = 25.0;
	string fileName = "camera_video.avi";
	write.open(fileName , codec ,fps ,img.size() ,isColorChannel);
	if (!write.isOpened()) {
		cout << "camera stream fialure." << endl;
		return;
	}
	while(1) {
		if (!video.read(img)) {
			cout << "camera connection break." << endl;
			break;
		}
		write.write(img);
		//write << img;
		imshow("Camera" ,img);
		char c = waitKey(50);
		if (c == 27) {
			break;//ESC
		}

	}
	video.release();
	write.release();
	return;
}

int main(int argc ,char ** argv) {
	videoWrite();	
	return 0;
}
