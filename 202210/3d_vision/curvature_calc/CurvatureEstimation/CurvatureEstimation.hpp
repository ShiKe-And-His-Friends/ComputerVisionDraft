#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#include "InvalidData.hpp"

void LineEstimation(cv::Mat rangeImage, cv::Vec3f iCenter, std::vector<cv::Point3f>& lines_coordinate ,float angle);


double  getArcCurvity(std::vector<cv::Point3f>& curve); //���׹�ʽ���㻡������ //https ://it.cha138.com/nginx/show-299041.html