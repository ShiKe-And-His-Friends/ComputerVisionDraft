/*
	camera clibration all photos
	*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "ClibaHelp.h"

// 测试读写文件格式
void TestSaveAndRead() {
	std::cout << "验证文件读写" << std::endl;

	cv::Mat mat(640, 480, CV_16UC1);

	for (int i = 0; i < mat.rows; i++) {
		unsigned short* ptr = mat.ptr<unsigned short>(i);
		for (int j = 0; j < mat.cols; j++) {
			ptr[j] = static_cast<unsigned short>((i*20 + j) % 128);
		}
	}

	cv::String name = "a.tiff";
	cv::imwrite(name, mat);
	cv::Mat mat2 = cv::imread(name ,cv::IMREAD_UNCHANGED);

	std::cout << mat.type() << std::endl;
	std::cout << mat2.type() << std::endl;

	cv::imshow("mat1", mat);
	cv::imshow("mat2", mat2);
	cv::waitKey(0);

}

int main(int argc, char** argv) {

	// 读取文件夹下的所有孔洞图片的目录
	ClibaHelp* clibaHelp = new ClibaHelp;
	vector<String> photos_dirs;
	string circle_Photo_Dir1 = "..//CameraData//Ignore_images";
	
	clibaHelp->getAllFileFromDirctory(circle_Photo_Dir1 ,photos_dirs ,0);
	if (photos_dirs.size() <= 0) {
		cout << "No photos";
		return -1;
	}

	// 读取所有孔洞图片
	vector<Mat> photos;
	vector<String>::iterator it_photos_dirs, end_photos_dirs;
	it_photos_dirs = photos_dirs.begin();
	end_photos_dirs = photos_dirs.end();

	for (; it_photos_dirs != end_photos_dirs; it_photos_dirs++) {
		Mat circle_Photo = imread((*it_photos_dirs), IMREAD_UNCHANGED);
		photos.push_back(circle_Photo);
	}
	cout << "Photos Size " << photos.size() << endl;

	// 找图的特征角点
	int return_flag_values = 0;
	int flags = CALIB_CB_SYMMETRIC_GRID;
	Size patternSize(14, 13);
	Size minSize(5, 5);
	vector<vector<Point2f>> circle_Photo_Corners;
	TermCriteria termCriteria(TermCriteria::COUNT | TermCriteria::EPS, 0, 0.001);

	vector<Mat>::iterator it_photos, end_photos;
	it_photos = photos.begin();
	end_photos = photos.end();

	for (; it_photos != end_photos; it_photos++) {
		return_flag_values++;

		vector<Point2f> circle_Photo_Corner_Sub;
		bool found = findCirclesGrid((*it_photos), patternSize, circle_Photo_Corner_Sub, flags);
		if (found) {
			cout << "found circle[" << return_flag_values << "]" << endl;
			cornerSubPix((*it_photos), circle_Photo_Corner_Sub, minSize, Size(-1, -1), termCriteria);
		}
		else {
			cout << "no found circle[" << return_flag_values << "]" << endl;
			return -return_flag_values;
		}
		circle_Photo_Corners.push_back(circle_Photo_Corner_Sub);

		// 画图看一下标定效果
		if (clibaHelp->DEBUG_SWITCH) {
			drawChessboardCorners((*it_photos), patternSize, circle_Photo_Corner_Sub, found);
			imshow("cricle photo" + return_flag_values, (*it_photos));
			waitKey(800);
			destroyWindow("cricle photo" + return_flag_values);
		}
	}

	// 坐标轴
	vector<Point3f> circle_Photo_Axis;
	clibaHelp->calcChessboards(patternSize, circle_Photo_Axis);

	// 计算相机内参、外参
	Mat cameraMatrix, distCoeffs;
	clibaHelp->runClibration(photos, circle_Photo_Corners, patternSize, cameraMatrix, distCoeffs);

	// 保存相机畸变参数
	const string camera_matrix_distort_coffes_dir = "..//CameraData//20220710_camera_matrix_distort_coffes.yaml";
	FileStorage file(camera_matrix_distort_coffes_dir ,FileStorage::WRITE);
	file << "Author" << "sk95120";
	file << "Data" << "None";
	file << "Camera Matrix" << cameraMatrix;
	file << "Distort Coffes" << distCoeffs;
	file.release();

	// 释放
	delete clibaHelp;

	return 0;
}
