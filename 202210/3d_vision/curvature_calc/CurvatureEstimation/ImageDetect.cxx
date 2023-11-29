#include "ImageDetect.hpp"

cv::Mat convert32Fto8U(cv::Mat rawImage){

	if (rawImage.channels() != 1) {
		std::cout << "Raw Image Channel Error." << std::endl;
		exit(-1);
	}
	if (rawImage.type() != CV_32F) {
		std::cout << "Raw Image Type Error." << rawImage.type() << std::endl;
		exit(-1);
	}

	int width = rawImage.cols;
	int height = rawImage.rows;

	cv::Mat dstMat(height, width, CV_8UC1, cv::Scalar(0));

	float max = std::numeric_limits<float>::min();
	float min = std::numeric_limits<float>::max();

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			float value = rawImage.at<float>(j, i);
			if (value != INVALID_POINTS) {
				if (value < min) {
					min = value;
				}
				if (value > max) {
					max = value;
				}
			}
		}
	}

	//std::cout << "max " << max << " min " << min << std::endl;
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			float value = rawImage.at<float>(j, i);
			if (value != INVALID_POINTS) {
				unsigned char convert_value = 0;
				convert_value = static_cast<unsigned char>((value - min) / (max - min) * 255);
				dstMat.at<unsigned char>(j, i) = convert_value;
			}
		}
	}


	return dstMat;
}


// 计算点到圆心的距离
double distance_calc(const std::vector<double>& point, const std::vector<double>& center) {
	return std::sqrt((point[0] - center[0]) * (point[0] - center[0]) + (point[1] - center[1]) * (point[1] - center[1]));
}

// 最小二乘法拟合圆
std::vector<double> leastSquaresCircle(const std::vector<std::vector<double>>& points) {
	// 初始化圆心和半径
	std::vector<double> center(2, 0);
	double radius = 0;

	// 计算所有点到当前圆心的距离的平方和
	double sumDistances = 0;

	// 迭代计算最小二乘拟合圆
	for (int i = 0; i < points.size(); i++) {
		double distance = distance_calc(points[i], center);
		sumDistances += distance * distance;

		// 更新圆心和半径
		center[0] += distance * points[i][0] / sumDistances;
		center[1] += distance * points[i][1] / sumDistances;
		radius = std::sqrt(sumDistances / points.size());
	}

	std::cout << "拟合圆的半径: " << radius << std::endl;
	return center;
}

/***************************************     RANSAC 拟合圆   ********************************************************/
