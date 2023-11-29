#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "InvalidData.hpp"

// sincevision 深度图转 灰度图
cv::Mat convert32Fto8U(cv::Mat rawImage); 

// 伪-最小平方法拟合
std::vector<double> leastSquaresCircle(const std::vector<std::vector<double>>& points);

