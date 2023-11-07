#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/Xfeatures2d/features2d.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	// 待匹配的两幅图像，其中img1包括img2，也就是在img1中识别处img2
	Mat img1 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.jpg", -1);
	Mat img2 = cv::imread("..//..//20231101_ImgFeatureDetectionDescription//box_img1.jpg", -1);

	if (!img1.data || !img2.data) {
		std::cout << "no photo." << std::endl;
		return -1;
	}

	SIFT sift1 ,sift2; //实例化SIFT类

	vector<KeyPoint> key_points1 ,key_points2; //特征点
	// descriptors 为描述符 mascara为掩码矩阵
	Mat descriptors1, descriptors2, mascara;

	sift(img1, mascara, key_points1, descriptors1); // 执行SIFT运算
	sift(img2, mascara, key_points2, descriptors2);

	//实例化暴力匹配器-BruteForceMatcher
	BruteForceMatcher<L2 <float>> matcher;
	//定义匹配器算子
	vector<DMatch> matches;
	//实现描述符之间的匹配，得到算子matches
	matcher.match(descriptors1 ,descriptors2 ,matches);

	// 提取前30个最佳匹配结果
	std::nth_element(matches.begin(),
		matches.begin()+29,
		matches.end());
	//剔除掉其余的匹配结果
	matches.erase(matches.begin()+30 ,matches.end());

	//显示
	namedWindow("SIFT_matches", WINDOW_AUTOSIZE);
	Mat img_matches;
	// 在输出图像中绘制特征点
	drawMatches(
		img1,key_points1, //第一幅图的特征点矢量
		img2, key_points2, //第二幅图的特征点矢量
		matches, //匹配器算子
		img_matches,//匹配输出图像
		Scalar(255 ,255,255), //用白色直线连接两幅图像中的特征点
		//以特征点为中心画圆，圆的半径表示特征点的大小，直线表示特征点的方向
		);

	imshow("SIFT_matches", img_matches);
	waitKey(2000);
	imwrite("..//..//20231101_ImgFeatureDetectionDescription//result//SIFT_matches.png", img_matches);

	return 0;
}