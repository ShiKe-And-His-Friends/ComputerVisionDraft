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


// ����㵽Բ�ĵľ���
double distance_calc(const std::vector<double>& point, const std::vector<double>& center) {
	return std::sqrt((point[0] - center[0]) * (point[0] - center[0]) + (point[1] - center[1]) * (point[1] - center[1]));
}

// ��С���˷����Բ
std::vector<double> leastSquaresCircle(const std::vector<std::vector<double>>& points) {
	// ��ʼ��Բ�ĺͰ뾶
	std::vector<double> center(2, 0);
	double radius = 0;

	// �������е㵽��ǰԲ�ĵľ����ƽ����
	double sumDistances = 0;

	// ����������С�������Բ
	for (int i = 0; i < points.size(); i++) {
		double distance = distance_calc(points[i], center);
		sumDistances += distance * distance;

		// ����Բ�ĺͰ뾶
		center[0] += distance * points[i][0] / sumDistances;
		center[1] += distance * points[i][1] / sumDistances;
		radius = std::sqrt(sumDistances / points.size());
	}

	std::cout << "���Բ�İ뾶: " << radius << std::endl;
	return center;
}

/***************************************     RANSAC ���Բ   ********************************************************/
