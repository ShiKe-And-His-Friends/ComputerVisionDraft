#include <iostream>
#include <fstream>
#include <cmath> 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/********************************************

	2023/05/09 武志凯画图
	
	step_y_index  // 起始坐标
	step_on_gray_value  // 起始亮度
	step_off_gray_value  // 对称减小亮度
	x_start //起始x坐标
	x_end  //结束x坐标
********************************************/
void draw_symmetry_canves(Mat photo ,int step_y_index ,int step_on_gray_value ,int step_off_gray_value ,int x_start ,int x_end);

// 起始坐标偏移1
#define INDEX_VALUE 1


/********************************************

	霍夫圆检测

********************************************/
int main2() {

	Mat img, gray;

	img = imread("E:/project/Calib_Algorithm/TestData/20230407_gride_crile/Intensity/indesty.bmp", 1);
	//if (argc != 2 || !(img = imread(argv[1], 1)).data)
	//	return -1;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	// smooth it, otherwise a lot of false circles may be detected
	//GaussianBlur(gray, gray, Size(9, 9), 2, 2);
	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT,
		1 ,250 ,10 ,30 ,60);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		std::cout << "Circle index " << i << std::endl;
	}
	namedWindow("circles", 1);
	imshow("circles", img);
	imwrite("circles.png" ,img);
	waitKey(0);

	return 0;
}

/************** 加载range image *****************/
int range_image_info() {
	int w = 4320;
	int h = 1000;
	
	unsigned char *data = new unsigned char[4 * w * h];
	memset(data ,0 ,sizeof(float) * w * h);
	ifstream in("E:/project/LCubor3D_LST_Algorithm/LCubor3D_LST_TestData/yml_temp/LineScan_rangimg.dat", std::ios::in | std::ios::trunc );
	in.read((char*)data ,sizeof(float) * w * h);
	in.close();

	cv::Mat rangeTempImg(h, w, CV_32FC1);
	memcpy(rangeTempImg.data, data, 4 * w * h);

	cv::imshow("a" , rangeTempImg);
	cv::waitKey(0);

	//for (int j = 0; j < h; j++) {
	//
	//	for (int i = 0; i < w; i++) {
	//		int index = j * w + i;
	//		unsigned char value = data[index];
	//		std::cout << static_cast<int>(value) << " ";
	//	}
	//}

	return 0;
}

int main_draw_lines() {

	std::cout << "save photo..." << std::endl;

	Mat photo(1088 ,4320 ,CV_8UC1 ,Scalar(8));

	/*******************************************************************
		第1台阶
		列[100 112]   行[1 24]   值[60 ,180 ,60]
	*******************************************************************/
	draw_symmetry_canves(photo ,100 ,60 ,180 ,1 ,24 * 18);

	/*******************************************************************
		第2台阶
		列[200 212]   行[26 52]   值[80 ,200 ,80]
	*******************************************************************/
	draw_symmetry_canves(photo, 200, 80, 200, 26 * 18 + 1 , 52 * 18);

	/*******************************************************************
		第3台阶
		列[300 312]   行[54 79]   值[100 ,220 ,100]
	*******************************************************************/
	draw_symmetry_canves(photo, 300, 100, 220, 54 * 18 + 1, 79 * 18);

	/*******************************************************************
		第4台阶
		列[400 412]   行[81 107]   值[120 ,240 ,120]
	*******************************************************************/
	draw_symmetry_canves(photo, 400, 120, 240, 81 * 18 + 1, 107 * 18);

	/*******************************************************************
		第5台阶
		列[539 550]   行[109 134]   值[140 ,260,140]
	*******************************************************************/
	draw_symmetry_canves(photo, 538, 140, 260, 109 * 18 + 1, 134 * 18);

	/*******************************************************************
		第6台阶
		列[677 689]   行[140 164]   值[120 ,240 ,120]
	*******************************************************************/
	draw_symmetry_canves(photo, 677, 120, 240, 140 * 18 + 1, 164 * 18);

	/*******************************************************************
		第7台阶
		列[777 789]   行[165 188]   值[100 ,220 ,100]
	*******************************************************************/
	draw_symmetry_canves(photo, 777, 100, 220, 165 * 18 + 1, 188 * 18);

	/*******************************************************************
		第8台阶
		列[877 889]   行[190 213]   值[80 ,200 ,80]
	*******************************************************************/
	draw_symmetry_canves(photo, 877, 80, 200, 190 * 18 + 1, 213 * 18);

	/*******************************************************************
		第9台阶
		列[977 989]   行[223 239]   值[60 ,180 ,60]
	*******************************************************************/
	draw_symmetry_canves(photo, 977, 60, 180, 215 * 18 + 1, 239 * 18);

	// 保存图片到本地
	imwrite("line_center.png" ,photo);

	return 0;
}

/****
	志凯要的第二图，验证连续性优先
***/
int main_make_line() {

	std::cout << "save photo..." << std::endl;

	Mat photo(1088, 4320, CV_8UC1, Scalar(8));

	/*******************************************************************
		第1台阶
		列[100 112]   行[1 24]   值[60 ,180 ,60]
		
		draw_symmetry_canves(photo, 100, 60, 180, 1, 24 * 18);
	*******************************************************************/
	
	// 两条一样的亮度的光条，放下面
	draw_symmetry_canves(photo, 1000, 60, 180, 1, 10);
	draw_symmetry_canves(photo, 1016, 60, 180, 1, 10);

	// 两条一样的亮度的光条，放中间
	draw_symmetry_canves(photo, 508, 60, 180, 12, 22);
	draw_symmetry_canves(photo, 523, 80, 200, 12, 22);
	draw_symmetry_canves(photo, 538, 100, 220, 12, 32);
	draw_symmetry_canves(photo, 553, 60, 180, 12, 22);
	draw_symmetry_canves(photo, 568, 80, 200, 12, 22);

	// 距离权重
	draw_symmetry_canves(photo, 514, 60, 180, 33, 43);
	draw_symmetry_canves(photo, 563, 60, 180, 33, 43);

	// 基线2
	draw_symmetry_canves(photo, 250, 80, 200, 47, 57);

	// 距离权重2
	draw_symmetry_canves(photo, 231, 60, 180, 58, 68);
	draw_symmetry_canves(photo, 270, 60, 180, 58, 68);

	// 基线3
	draw_symmetry_canves(photo, 250, 80, 200, 68, 78);

	// 距离权重3
	draw_symmetry_canves(photo, 230, 60, 180, 79, 89);
	draw_symmetry_canves(photo, 269, 60, 180, 79, 89);

	// 基线4
	draw_symmetry_canves(photo, 250, 80, 200, 99, 109);

	// 灰度权重1
	draw_symmetry_canves(photo, 230, 100, 220, 109, 119);
	draw_symmetry_canves(photo, 270, 60, 180, 109, 119);

	// 基线5
	draw_symmetry_canves(photo, 250, 80, 200, 119, 129);

	// 灰度权重5
	draw_symmetry_canves(photo, 230, 60, 180, 129, 139);
	draw_symmetry_canves(photo, 270, 100, 220, 129, 139);

	// 基线6
	draw_symmetry_canves(photo, 250, 80, 200, 149, 159);

	// 灰度和距离权重6
	draw_symmetry_canves(photo, 220, 100, 220, 159, 169);
	draw_symmetry_canves(photo, 270, 60, 180, 159, 169);

	// 基线7
	draw_symmetry_canves(photo, 250, 80, 200, 172, 182);

	// 灰度和距离权重7
	draw_symmetry_canves(photo, 150, 100, 220, 182, 192);
	draw_symmetry_canves(photo, 200, 80, 200, 182, 192);
	draw_symmetry_canves(photo, 300, 100, 220, 182, 192);
	draw_symmetry_canves(photo, 350, 80, 200, 182, 192);

	// 保存图片到本地
	imwrite("D:/01Work_Plane_Groupby_Data/202305/武志凯要图/连续性优先/20230613_line_center.png", photo);

	return 0;
}


void draw_symmetry_canves(Mat photo,int step_y_index, int step_on_gray_value, int step_off_gray_value, int x_start, int x_end) {
	int STEPON_VALUE = 20;
	for (int times = 0; times < 6; times++) {
		unsigned char* dataPtr1 = photo.ptr<unsigned char>(step_y_index + times - INDEX_VALUE); // 列
		for (int x = (x_start - INDEX_VALUE); x <= (x_end - INDEX_VALUE); x++) {
			dataPtr1[x] = static_cast<unsigned char>(step_on_gray_value);
		}
		step_on_gray_value += STEPON_VALUE;
	}

	for (int times = 0; times < 7; times++) {
		unsigned char* dataPtr2 = photo.ptr<unsigned char>(step_y_index + times + 6 - INDEX_VALUE); // 列
		for (int x = (x_start - INDEX_VALUE); x <= (x_end - INDEX_VALUE); x++) {
			dataPtr2[x] = static_cast<unsigned char>(step_off_gray_value > 255 ? 255 : step_off_gray_value);
		}
		step_off_gray_value -= STEPON_VALUE;
	}

}


//标记图片中的红色点和蓝色点
int main_mark_red_and_blue() {

	std::cout << " DRAW CENTER POINTS " << std::endl;

	//RGB 颜色
	//

	// 加载图片
	cv::Mat gray_image = cv::imread("D:/01Work_Plane_Groupby_Data/202305/20230614_中心灰度值的权重方案验证/样图1_欠曝/4_mask.png", 0);


	// 彩色图片
	cv::Mat color_image;
	cv::cvtColor(gray_image, color_image, COLOR_GRAY2RGB);

	// 读取两个方案的亚像素精度
	std::ifstream in_symmetric_light_center_point("D:/01Work_Plane_Groupby_Data/202305/20230614_中心灰度值的权重方案验证/样图1_欠曝/delta1_31x1.a"); //blue
	std::ifstream in_gradient_light_center_point("D:/01Work_Plane_Groupby_Data/202305/20230614_中心灰度值的权重方案验证/样图1_欠曝/gaussian_31x1.a"); //二值化正确的阈值20 // red


	int num_symmetric = -1;
	int num_gradient = -1;
	int index = 0;

	index = 0;

	int* sum_count = new int[64];
	memset(sum_count, 0, sizeof(int) * 64);

	while (in_gradient_light_center_point >> num_gradient && in_symmetric_light_center_point >> num_symmetric) {
		//std::cout << num_gradient << " " << num_symmetric << std::endl;

		if (num_symmetric == num_gradient) {

			if (num_symmetric != 0) {
				cv::Point2d point1(index, num_symmetric);
				cv::circle(color_image, point1, 0, cv::Scalar(0, 255, 0), 1);
				sum_count[0] ++;
			}

		}
		else {
			if (num_symmetric != 0) {
				cv::Point2d point1(index, num_symmetric);
				cv::circle(color_image, point1, 0, cv::Scalar(250, 51, 253), 1);
			}

			if (num_gradient != 0) {
				cv::Point2d point2(index, num_gradient);
				cv::circle(color_image, point2, 0, cv::Scalar(0, 255, 255), 1);
			}
			int div = std::abs(num_symmetric - num_gradient);

			if (div < 64) {
				sum_count[div] ++;
			}
		}

		index++;
	}

	for (int i = 0; i < 64; i++) {
		if (sum_count[i] != 0) {
			std::cout << "差值" << i << " ";
			std::cout << sum_count[i] << "个数。" << std::endl;
		}
	}

	// 保存图片
	cv::imwrite("D:/01Work_Plane_Groupby_Data/202305/20230614_中心灰度值的权重方案验证/样图1_欠曝/4_mask_delta1对称性方法对比原高斯方案_31x1.png", color_image);

	return 0;
}


// 标记图片中的单个点
int main_mark_red() {

	std::cout << " DRAW CENTER POINTS " << std::endl;

	//RGB 颜色
	//

	// 加载图片
	cv::Mat gray_image = cv::imread("D:/01Work_Plane_Groupby_Data/202305/武志凯要图/2023_06_13_连续性优先3/20230613_line_center_masked.png", 0);


	// 彩色图片
	cv::Mat color_image;
	cv::cvtColor(gray_image, color_image, COLOR_GRAY2RGB);

	// 读取两个方案的亚像素精度
	std::ifstream in_symmetric_light_center_point("D:/01Work_Plane_Groupby_Data/202305/武志凯要图/2023_06_13_连续性优先3/20230613_line_center_中心像素点_2.a"); //blue
	
	int num_symmetric = -1;
	int num_gradient = -1;
	int index = 0;

	index = 0;

	int* sum_count = new int[64];
	memset(sum_count, 0, sizeof(int) * 64);

	while (in_symmetric_light_center_point >> num_symmetric) {
		//std::cout << num_gradient << " " << num_symmetric << std::endl;

		if (num_symmetric != 0) {

			num_symmetric--;
			cv::Point2d point1(index, num_symmetric);
			cv::circle(color_image, point1, 0, cv::Scalar(0 , 0, 254), 1);
		}


		index++;
	}

	// 保存图片
	cv::imwrite("D:/01Work_Plane_Groupby_Data/202305/武志凯要图/2023_06_13_连续性优先3/20230613_line_center_color_2.png", color_image);

	return 0;
}

// 单应矩阵转化后的结果
int homography_calc_convert() {

	std::cout << "单应矩阵转换前后的坐标位置..." << std::endl;

	std::ifstream in_file_before_undistort("D:01Work_Plane_Groupby_Data/202306/验证单应矩阵的变换效果/对标定坐标的未处理的标定点_24张图片.a", std::ios::in | std::ios::binary);
	std::ifstream in_file_after_undistort("D:01Work_Plane_Groupby_Data/202306/验证单应矩阵的变换效果/对标定坐标的处理后的标定点_24张图片.a", std::ios::in | std::ios::binary);

	float index_x_before_undistort, index_y_before_undistort , index_x_after_undistort, index_y_after_undistort;
	
	// 单应矩阵信息
	cv::Mat range_mat(1088 ,4320, CV_8UC3,cv::Scalar(0));
	std::vector<float> vec_x_before_undistort; //全图畸变矫正前的点
	std::vector<float> vec_y_before_undistort;
	std::vector<float> vec_x_after_undistort; //全图畸变矫正后的点
	std::vector<float> vec_y_after_undistort;

	for (int i = 0; i < 1225; i++) {
		in_file_before_undistort >> index_x_before_undistort >> index_y_before_undistort;
		in_file_after_undistort >> index_x_after_undistort >> index_y_after_undistort;
	
		vec_x_before_undistort.push_back(index_x_before_undistort);
		vec_y_before_undistort.push_back(index_y_before_undistort);
		vec_x_after_undistort.push_back(index_x_after_undistort);
		vec_y_after_undistort.push_back(index_y_after_undistort);

		//画出坐标点
		cv::Point2d point2(index_x_before_undistort, index_y_before_undistort);
		cv::circle(range_mat, point2, 0, cv::Scalar(0, 0, 255), 6);

		cv::Point2d point3(index_x_after_undistort, index_y_after_undistort);
		cv::circle(range_mat, point3, 0, cv::Scalar(0, 255, 0), 6);

	}

	//单应矩阵转换前的点
	cv::imwrite("D:/01Work_Plane_Groupby_Data/202306/验证单应矩阵的变换效果/单应矩转换的坐标点_对标定坐标的处理后的标定点_24张图片.png", range_mat);

	//单应矩阵 20.5度的托架
	float hh[9] = { 9.97895282e-03 ,-1.69434561e-03  ,-9.46640193e-01  ,1.84434739e-05
	   ,1.41190048e-02  ,-7.36118317e-01  ,-1.92809892e-07 ,-9.71286281e-05 ,1};

	//单应矩阵 测试正方体
	//float hh[9] = { 9.98351444e-03 ,-1.69151474e-03  ,-9.50107694e-01 ,5.03275296e-05
	//   ,3.98502909e-02 ,-2.07852006e+00 ,-9.87895348e-08 ,-9.68309760e-05 ,1.0 };
	
	std::vector<float> vec_u_before_undistort;//全图畸变矫正前的实际尺寸点
	std::vector<float> vec_v_before_undistort;
	std::vector<float> vec_u_after_undistort;//全图畸变矫正后的实际尺寸点
	std::vector<float> vec_v_after_undistort;

	for (int i = 0; i < 1225; i++) {
		float x = vec_x_before_undistort[i];
		float y = vec_y_before_undistort[i];

		//std::cout << x << " " << y << std::endl;

		float p_inv = 1.f/ (x * hh[6] + y * hh[7] + hh[8]);
		float u = (x * hh[0] + y * hh[1] + hh[2]) * p_inv; 
		float v = (x * hh[3] + y * hh[4] + hh[5])* p_inv;

		// std::cout << u << " " << v << std::endl;

		//便于绘制图片
		u = u + 1;
		v = v + 1;
		u = u * 100;
		v = v * 100;

		vec_u_before_undistort.push_back(u);
		vec_v_before_undistort.push_back(v);

		x = vec_x_after_undistort[i];
		y = vec_y_after_undistort[i];

		p_inv = 1.f / (x * hh[6] + y * hh[7] + hh[8]);
		u = (x * hh[0] + y * hh[1] + hh[2]) * p_inv;
		v = (x * hh[3] + y * hh[4] + hh[5]) * p_inv;

		//便于绘制图片
		u = u + 1;
		v = v + 1;
		u = u * 100;
		v = v * 100;

		vec_u_after_undistort.push_back(u);
		vec_v_after_undistort.push_back(v);

	}

	//保存标定过后的图片
	cv::Mat object_mat(1800 ,4200 ,CV_8UC3 ,cv::Scalar(0));
	for (int i = 0; i < 1225; i++) {
		float u = vec_u_before_undistort[i];
		float v = vec_v_before_undistort[i];

		float u2 = vec_u_after_undistort[i];
		float v2 = vec_v_after_undistort[i];

		cv::Point2d point2(u, v);
		cv::circle(object_mat, point2, 0, cv::Scalar(0, 0, 255), 9);

		cv::Point2d point3(u2, v2);
		cv::circle(object_mat, point3, 0, cv::Scalar(0, 255, 0), 9);
	}
	cv::imwrite("D:/01Work_Plane_Groupby_Data/202306/验证单应矩阵的变换效果/单应矩转换转换后的坐标点_对标定坐标的处理后的标定点_24张图片.png" , object_mat);

	return 0;
}


// 计算线性度的方法
//void drlError()
int main()
{
	std::cout << "计算线性度的方法..." << std::endl;

	std::ifstream in_file("D:/01Work_Plane_Groupby_Data/202306/验证单应矩阵的变换效果/temp_可删除.a", std::ios::in | std::ios::binary);

	float x, y;

	vector<float> vecValsMeasure_x;
	vector<float> vecValsMeasure_y;
	double resLinear;

	//拟合直线
	int iSize = 9; //35
	cv::Mat matA(iSize, 2, CV_32FC1, cv::Scalar(0));
	cv::Mat matB(iSize, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat matX;
	
	for (int i = 0; i < iSize; i++)
	{
		in_file >> x >> y;
		std::cout << x << " " << y << std::endl;
		vecValsMeasure_x.push_back(x);
		vecValsMeasure_y.push_back(y);
	}
	
	for (int i = 0; i < iSize; i++)
	{
		matA.at<float>(i, 0) = vecValsMeasure_x[i];
		matA.at<float>(i, 1) = 1;
		matB.at<float>(i, 0) = vecValsMeasure_y[i];
	}
	if (!cv::solve(matA, matB, matX, cv::DECOMP_SVD))
	{
		return 0;
	}

	std::cout << "Mat X " << matX << std::endl;

	//计算线性度
	float fA = matX.ptr<float>(0)[0];
	float fB = matX.ptr<float>(1)[0];
	float fDist = 0.f;
	vector<float> vecDists;

	std::cout << "a " << fA << " b " << fB << std::endl;

	for (int i = 0; i < iSize; i++)
	{
		fDist = abs(fA * vecValsMeasure_x[i] + fB - vecValsMeasure_y[i] );
		vecDists.emplace_back(fDist);
	}
	double dMax = cv::norm(vecDists, cv::NORM_INF);

	double dRange = static_cast<double>(abs(vecValsMeasure_y[iSize - 1] - vecValsMeasure_y[0]));
	resLinear = (dMax / dRange) * 100;	//单位 %

	std::cout << "dMax " << dMax << " dRange " << dRange << std::endl;

	std::cout << "##########################################" << std::endl;
	std::cout << "### fitLine" << std::endl;

	// 加载点
	std::vector<cv::Point> points;
	for (int i = 0; i < iSize; i++) {
		points.push_back(cv::Point(vecValsMeasure_x[i], vecValsMeasure_y[i]));
		// std::cout << " " << (i+1) << " " << vecValsMeasure[i] << std::endl;
	}

	cv::Vec4f result;
	cv::fitLine(points, result, cv::DIST_WELSCH, 0, 0.001, 0.001);
	std::cout << " 直线起点 (x0 ,y0) " << result[2] << " " << result[3] << std::endl;
	std::cout << " 直线斜率 k " << (result[1] / result[0]) << "  vx " << result[0] << " vy " << result[1] << std::endl;


	std::cout << "线性度 " << (resLinear) << "%" << std::endl;

	return 0;
}
