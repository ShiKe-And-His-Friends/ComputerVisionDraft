#include<iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

int main() {

	std::cout << "Circle Fitiing..." << std::endl;

	//����Բ��ͼƬ
	cv::Mat src = cv::imread("E:/project/EDCircle/Image/HoughCricles2_Canny.bmp", -1);
	cv::imwrite("C://Users//s05559//Desktop//inliners_photo//src.bmp",src);
	if (!src.data) {
		std::cout << "no photo..." << std::endl;
		return - 1;
	}
	if (src.type() != CV_8UC1) {
		std::cout << "format error..." << std::endl;
		return -1;
	}

	//Ѱ��������
	std::vector<cv::Point2f> sample_points;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			float value = src.at<unsigned char>(j ,i);
			if (value == 255) {
				sample_points.push_back(cv::Point2f(i ,j));
			}
		}
	}
	std::cout << "����������� " << sample_points.size() << std::endl;

	int all_size = sample_points.size();

	//���ѡ�����ĵ�
	int interator = 2500;
	int min_inliner_limit = 700;
	int best_point_size = -INT_MAX - 1;
	double outliner_mean_bias = 1e10;
	int spac_radius_threhold = 7;
	std::vector<int> index_(all_size);
	for (int kk = 0; kk < index_.size(); kk++) {
		index_[kk] = kk;
	}
	std::vector<int> inliners_; //�ڵ�����
	std::vector<double> space_radius_bias; //���
	int best_point_a = -INT_MAX - 1;
	int best_point_b = -INT_MAX - 1;
	int best_point_c = -INT_MAX - 1;

	for (int i = 0; i < interator; i++) {

		//�����������
		for (int k = 0; k < 1000; k++) {
			for (int kk = 0; kk < all_size - 1; kk++) {
				srand(all_size * 2 + kk);
				std::swap(index_[kk], index_[kk + rand() % (all_size - kk)]);
			}
		}
		//��ѡ3����Ե��
		cv::Point2f a_point, b_point, c_point;
		a_point = sample_points[index_[0]];
		b_point = sample_points[index_[1]];
		c_point = sample_points[index_[2]];

		//����Բ�ĺͰ뾶
		float a = 2 * (b_point.x - a_point.x), b = 2 * (b_point.y - a_point.y), c = b_point.x * b_point.x + b_point.y * b_point.y - a_point.x * a_point.x - a_point.y * a_point.y;
		float d = 2 * (c_point.x - b_point.x), e = 2 * (c_point.y - b_point.y), f = c_point.x * c_point.x + c_point.y * c_point.y - b_point.x * b_point.x - b_point.y * b_point.y;

		float x = (b * f - e * c) / (b * d - e * a);
		float y = (d * c - a * f) / (b * d - e * a);
		float r = sqrt((x - a_point.x) * (x - a_point.x) + (y - a_point.y) * (y - a_point.y));

		//���Բ������Բ�ĵ�
		if (r > src.rows || r > src.cols) {
			continue;
		}

		//������ϵ�
		inliners_.clear();
		space_radius_bias.clear();
		for (int h = 3; h < all_size; h++) {
			cv::Point2f point = sample_points[index_[h]];
			float radius = std::sqrt((point.x - x) * (point.x - x) + (point.y - y) * (point.y - y));
			if (std::fabs(radius - r) < spac_radius_threhold) {
				inliners_.push_back(index_[h]);
			}
			else {
				space_radius_bias.push_back(radius - r);
			}
		}
		

		// ���������ٵ��ڵ�
		if (inliners_.size() < min_inliner_limit) {
			continue;
		}

		//����ƫִ�ı�׼��ֵ
		double sum = 0;
		for (int p = 0; p < space_radius_bias.size(); p++) {
			sum += space_radius_bias[p];
		}
		double mean = sum / space_radius_bias.size();

		double std_dev = 0;
		double std_sum = 0;
		//��׼��
		for (int p = 0; p < space_radius_bias.size(); p++) {
			std_sum += (space_radius_bias[p] - mean) * (space_radius_bias[p] - mean);
		}
		std_dev = std::sqrt(std_sum) / space_radius_bias.size() - 1;

		if (mean < 0) {
			mean = -mean;
		}

		
		//��������ֵҪ����
		if (outliner_mean_bias > mean) {
		
			std::cout << "bias mean " << outliner_mean_bias << "  old mean " << mean << std::endl;
			std::cout << " best " << best_point_size << " inliners " << inliners_.size() << std::endl;
					
			//if (best_point_size < (int)(inliners_.size())) {
			//	continue;
			//}
		
			// ��¼����Բ�ĺͰ뾶
			outliner_mean_bias = mean;
			best_point_size = inliners_.size();
			best_point_a = index_[0];
			best_point_b = index_[1];
			best_point_c = index_[2];

			//����һ��Բ
			cv::Mat dstImage;
			cv::cvtColor(src ,dstImage ,cv::COLOR_GRAY2BGR);
			for (int o = 0; o < inliners_.size(); o++) {
				cv::circle(dstImage, cv::Point2f(x, y), r, cv::Scalar(0,0,255), 2, 8);
			}

			cv::imwrite("C://Users//s05559//Desktop//inliners_photo//inliners_�ڵ�����" + std::to_string(best_point_size) + "_�뾶" + std::to_string(r) + "_Բ��" + std::to_string(x) + "," + std::to_string(y) + "_������" + std::to_string(outliner_mean_bias) +".bmp", dstImage);
			std::cout << "�ڵ�����" << best_point_size << std::endl;
			std::cout << "�뾶 " << r << " Բ�� " << x << " , " << y << std::endl;
			std::cout << std::endl;

			//cv::namedWindow("inliner_image", cv::WINDOW_NORMAL);
			//cv::imshow("inliner_image", dstImage);
			//cv::waitKey(1000);
		}
	}

	std::cout << "Circle Fitting done..." << std::endl;

	return 0;
}