#include <iostream>

#include "../opecv340/include/opencv2/core/core.hpp"
#include "../opecv340/include/opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

int main0(int argc, char** argv)
{
#if 0
	cv::Mat image;
	image = imread("E:/test data/001021.bmp");
	if (image.data == nullptr)                     
	{
		cout << "图片文件不存在" << endl;
	}
	else
	{
		//显示图片
		imshow("middle", image);
		waitKey(0);
	}

	// 输出图片的基本信息
	cout << "图像宽为：" << image.cols << "\t高度为：" << image.rows << "\t通道数为：" << image.channels() << endl;
#endif

	//cv::cvtColor(mat0, mat0, COLOR_GRAY2RGB);
	Mat src_image = imread("C:/Users/l05574/Desktop/src_image.bmp", 0);
	//对图像进行裁剪
	int w = src_image.cols;//图像宽度
	int h = src_image.rows;//图像长度
	int x_start = 0;//裁剪区域起始点x坐标
	int y_start = 383;//裁剪区域起始点y坐标
	int width = w;//裁剪区域长度
	int height = 362;//裁剪区域高度top, bottom, left, right,

	//裁剪区域
	cv::Rect area(x_start, y_start, width, height);//裁剪区域的矩形表示
	cv::Mat roi = src_image(area);
	cv::Rect area_thr(x_start, y_start, width/3, height);//裁剪区域的矩形表示
	cv::Mat roi_thr = src_image(area_thr);
	//cv::imwrite("C:/Users/l05574/Desktop/roi_image.bmp", roi);

	//新建画布-顶部区域
	cv::Mat top_image = cv::Mat::zeros(h, w, CV_8UC1);
	top_image.setTo(cv::Scalar(8, 0, 0));
	//设置画布绘制区域并复制
	cv::Rect roi_top = cv::Rect(0, 1, roi.cols, roi.rows);
	roi.copyTo(top_image(roi_top));
	//cv::imshow("top_image", top_image);
	cv::imwrite("C:/Users/l05574/Desktop/top_image.bmp", top_image);

	//新建画布-底部区域
	cv::Mat bottom_image = cv::Mat::zeros(h, w, CV_8UC1);
	bottom_image.setTo(cv::Scalar(8, 0, 0));
	//设置画布绘制区域并复制
	cv::Rect roi_bottom = cv::Rect(0, 724, roi.cols, roi.rows);
	roi.copyTo(bottom_image(roi_bottom));
	//cv::imshow("bottom_image", bottom_image);
	cv::imwrite("C:/Users/l05574/Desktop/bottom_image.bmp", bottom_image);

	//新建画布-中心区域
	cv::Mat center_image = cv::Mat::zeros(h, w, CV_8UC1);
	center_image.setTo(cv::Scalar(8, 0, 0));
	//设置画布绘制区域并复制
	cv::Rect roi_center = cv::Rect(0, 364, roi.cols, roi.rows);
	cv::Mat center_roi_image_1_5 = cv::Mat::zeros(362, w, CV_8UC1);
	cv::Mat center_roi_zero_1_5 = cv::Mat::zeros(362, w, CV_8UC1);
	cv::addWeighted(center_roi_zero_1_5, 0, roi, 1.5, 0, center_roi_image_1_5);
	center_roi_image_1_5.copyTo(center_image(roi_center));
	//cv::imshow("center_image", center_image);
	cv::imwrite("C:/Users/l05574/Desktop/center_image.bmp", center_image);

	//新建画布-合成图像
	cv::Mat merge_image = src_image.clone();
	//设置画布绘制区域并复制
	cv::Rect roi_center_merge = cv::Rect(0, 364, roi.cols, roi.rows);
	//上下全部融合
	//roi.copyTo(merge_image(roi_top));
	//roi.copyTo(merge_image(roi_bottom));
	//上下1/3融合
	cv::Rect roi_top_thr = cv::Rect(0, 1, roi_thr.cols, roi_thr.rows);
	cv::Rect roi_bottom_thr = cv::Rect(w / 3 * 2 -1, 724, roi_thr.cols, roi_thr.rows);
	roi_thr.copyTo(merge_image(roi_top_thr));
	roi_thr.copyTo(merge_image(roi_bottom_thr));
	//中心光条亮度1.5
	cv::Mat center_roi_image = cv::Mat::zeros(362, w, CV_8UC1);
	cv::Mat center_roi_zero = cv::Mat::zeros(362, w, CV_8UC1);
	cv::addWeighted(center_roi_zero, 0, roi, 1.5, 0, center_roi_image);
	center_roi_image.copyTo(merge_image(roi_center_merge));

	//cv::imshow("merge_image", merge_image);
	cv::imwrite("C:/Users/l05574/Desktop/merge_image.bmp", merge_image);
	//waitKey(0);

	system("pause");
	return 0;
}
