#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "InvalidData.hpp"

// sincevision ���ͼת �Ҷ�ͼ
cv::Mat convert32Fto8U(cv::Mat rawImage); 

// α-��Сƽ�������
std::vector<double> leastSquaresCircle(const std::vector<std::vector<double>>& points);

