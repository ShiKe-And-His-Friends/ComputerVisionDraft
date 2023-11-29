#include "CurvatureEstimation.hpp"

#define PI 3.1415926

//计算直线斜率上的点的深度值
void LineEstimation(cv::Mat rangeImage, cv::Vec3f iCenter, std::vector<cv::Point3f>& lines_coordinate, float angle) {

	if (!rangeImage.data) {
		std::cout << "Raw Image No Data." << std::endl;
		return;
	}
	if (rangeImage.type() != CV_32F) {
		std::cout << "Raw Image Type Error." << std::endl;
		return;
	}
	if (angle > 360) {
		std::cout << "Angle Error." << std::endl;
		return;
	}

	//图片的边缘
	int height = rangeImage.rows;
	int widht = rangeImage.cols;

	//斜率
	float sin_value = sin(angle * PI / 180.0);
	float cos_value = cos(angle * PI / 180.0);
	float slope = sin_value / cos_value;

	std::cout << "slop " << slope << std::endl;

	//坐标
	float center_x = iCenter[0];
	float center_y = iCenter[1];
	float radius = iCenter[2];

	cv::Mat rangeData(rangeImage.rows, rangeImage.cols, rangeImage.type(), cv::Scalar(0));

	if (sin_value == 1 || sin_value == -1) {
		//斜率 不存在

		for (int j = center_y - radius; j < center_y + radius ; j ++) {
		
			if (j < 0 || j >= height) {
				continue;
			}

			//轮廓值
			float range = rangeImage.at<float>(j, center_x);
			if (range != INVALID_POINTS) {
				lines_coordinate.push_back(cv::Point3f(center_x,j, range));
			}
		}

	}
	else {
		for (int i = center_x - radius; i < center_x + radius; i++) {

			float y = center_y + slope * (i - center_x);
			float y1 = floor(y);

			float y2 = y1 + 1;

			if (i < 0 || i >= height) {
				continue;
			}
			if (y1 < 0 || y1 >= height) {
				continue;
			}
			if (y2 < 0 || y2 >= height) {
				continue;
			}

			//点不在圆内
			if (radius * radius < (y-center_y)*(y-center_y)+ (i-center_x)*(i-center_x)) {
				continue;
			}

			//轮廓值1
			float range1 = rangeImage.at<float>(y1, i);

			//轮廓值2 
			float range2 = rangeImage.at<float>(y2, i);;

			if (range1 != INVALID_POINTS && range2 != INVALID_POINTS) {
				float range = (y - y1) * range1 + (y2 - y) * range2;
				lines_coordinate.push_back(cv::Point3f(i ,y1, range));
			}
		}
	}
	
}


double disCv(const cv::Point2f& p1, const  cv::Point2f& p2)
{
	double dis = 0;
	dis = sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
	return dis;
}

//点 p 到线段（p1,p2）的距离-只计算三角形情况，且只计算垂直距离
//使用海伦公式
double p2Line(const cv::Point& p0, const cv::Point& p1, const  cv::Point& p2)
{
	double dis = 0;
	double a, b, c, p, S, h;
	a = disCv(p0, p1);
	b = disCv(p0, p2);
	c = disCv(p1, p2);
	p = (a + b + c) / 2;
	S = sqrt(p * (p - a) * (p - b) * (p - c));
	h = S * 2 / a;

	dis = h;
	return dis;
}

//计算弓形的曲率-此方法仍然只保证对连续点集确定有效
//计算方法：两端长度为弦长，高度为弓形高，以此计算曲率
double  getArcCurvity(std::vector<cv::Point3f>& curve)
{
	double curvity = 0;
	if (curve.size() < 4)
		return 0;

	cv::Point2f ps, pe;
	ps = cv::Point2f(curve[0].x , curve[0].z);
	pe = cv::Point2f(curve[curve.size()-1].x , curve[curve.size() - 1].z);
	double lArc = disCv(ps, pe);//弦长
	double hArc = 0;
	for (int i = 1; i < curve.size() - 1; ++i){
		double h = p2Line(cv::Point2f(curve[i].x , curve[i].z), ps, pe);//点到直线的距离
		if (hArc < h) {
			hArc = h;
		}
	}
	//曲率公式-使用R^2 = L^2 +(R-H)^2即 R= ( L^2/H +H)/2
	double R = 0.5 * (lArc * lArc / hArc + hArc);
	curvity = 1 / R;

	return curvity;
}