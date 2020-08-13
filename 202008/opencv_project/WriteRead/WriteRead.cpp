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

/**
 * YAML file
 * **/

void HandleYAML () {
	system("color F0");
	//string fileName = "data.xml";
	string fileName = "data.yaml";
	cv::FileStorage fwrite(fileName ,cv::FileStorage::WRITE);
	Mat mat = Mat::eye(3 ,3 ,CV_8U);
	fwrite.write("mat" ,mat);
	float x = 100;
	fwrite << "x" << x;
	string str = "Learn OpenCV4";
	fwrite << "str" << str;
	fwrite << "number_array" << "[" <<4<<5<<6<<"]";
	fwrite << "multi_nodes" << "{" << "month" << 8 << "day" << 13
		<< "year" << 2020 << "time"<< "[" << 0 << 1 << 2 << 3 << "]" << "}";
	fwrite.release();
	cv::FileStorage fread(fileName ,cv::FileStorage::READ);
	if (!fread.isOpened()) {
		cout << "open yaml file failure." << endl;
		return;
	} else {
		cout << "open yaml file success." << endl;
	}
	float XRead;
	fread["x"] >> XRead;
	cout << "x=" << XRead << endl;
	string strRead;
	fread["str"] >> strRead;
	cout << "str=" << strRead << endl;
	FileNode fileNode = fread["number_array"];
	cout << "number_array=[";
	for (FileNodeIterator i = fileNode.begin() ; i != fileNode.end() ; i++) {
		float a;
		*i >> a;
		cout << a << " ";
	}
	cout << "]" << endl;

	//read yaml date
	Mat matRead;
	fread["mat="] >> matRead;
	cout << "mat=" << mat << endl;
	FileNode fileNodeOne = fread["multi_nodes"];
	int month = (int)fileNodeOne["month"];
	int day = (int)fileNodeOne["day"];
	int year = (int)fileNodeOne["year"];
	cout << "multi_nodes:" <<endl;
	cout << " month = " << month << " day = " << day
		<< " year = " << year;
	cout << " time=[";
	for (int i = 0 ; i < 4 ; i++) {
		int a = (int)fileNodeOne["time"][i];
		cout << a << " ";
	}
	cout << "]" << endl;

	fread.release();

}
int main(int argc ,char ** argv) {
	HandleYAML();
	return 0;
}
