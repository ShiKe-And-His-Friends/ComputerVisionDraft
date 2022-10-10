/***
	
	识别 3D物体假设识别

	Global Hypothesis Verification  全局性假设

**/
#include <iostream>
#include <Eigen>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_based_signature.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZRBGA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType; //参考帧
typedef pcl::SHOT352 DescriptorType; // SHOT特征 (32 * 11 = 352)

// 点云风格  颜色 大小 结构体
struct CloudStyle {

	double r;
	double g;
	double b;
	double size;

	CloudStyle(double r ,double g ,double b ,double size):
		r(r),
		g(g),
		b(b),
		size(size){

	}
};


CloudStyle style_white(255.0, 255.0, 255.0, 4.0);
CloudStyle style_red(255.0 , 0, 0, 3.0);
CloudStyle style_green(0, 255.0, 0.0, 5.0);
CloudStyle style_cyan(93.0, 0.0, 217.0, 4.0);
CloudStyle style_violet(255.0, 0.0, 255.0, 8.0);

std::string model_filename_;
std::string scene_filename_;

// 算法参数 Algorithm params
bool show_keypoints_(false);
bool use_hough_(true);
float model_ss_(0.02f); // 模型点云下采样滤波搜索半径
float scene_ss_(0.02f); // 场景点云下采样滤波搜索半径
float rf_rad_(0.015f);
float descr_rad_(0.02f);
float cg_size_(0.01f); // 聚类 霍夫空间设置每个bin大小
float cg_thresh_(5.0f); // 聚类 阈值
int icp_max_iter_(5); //icp 最大迭代次数
float icp_corr_distance_(0.005f);
float hv_clutter_reg_(5.0f); // 模型假设验证 Hypothesis Verification
float hv_inlier_th_(0.005f);
float hv_occlusion_th_(0.01f);
float hv_rad_clutter_(0.03f);
float hv_regularizer_(3.0f);
float hv_rad_normals_(0.05f);
bool hv_detect_clutter_(true);

void showHelp(char *filename) {
	std::cout << "****************************************************************" << std::endl;
	std::cout << "	*" << std::endl;
	std::cout << "	*		Global Hypothese Verification Tutorial - Usage Guide" << std::endl;
	std::cout << "	*" << std::endl;
	std::cout << "****************************************************************" << std::endl;
	std::cout << "	Usage:" << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl;
	std::cout << "	Options:" << std::endl;
	std::cout << "		-h Show help" << std::endl;
	std::cout << "		-k Show keypoints" << std::endl;
	std::cout << "		--algorithm (Hough|GC)	| Clustering algorithm used (defalut Hough)" << std::endl;
	std::cout << "		--model_ss val			| Model uniform sampling radius (defalut "<< model_ss_ <<" )" << std::endl;
	std::cout << "		--scene_ss val			| Scene uniform sampling radius (defalut " << scene_ss_ << " )" << std::endl;
	std::cout << "		--rf_rad val			| Reference frame radius  (defalut " << rf_rad_ << " )" << std::endl;
	std::cout << "		--descr_rad val			| Descriptor radius (defalut " << descr_rad_ << " )" << std::endl;
	std::cout << "		--cg_size val			| Cluster size (defalut " << cg_size_ << " )" << std::endl;
	std::cout << "		--cg_thresh val			| Clustering thrshold (defalut " << cg_thresh_ << " )" << std::endl;
	std::cout << "		--icp_max_iter val		| ICP max iterations number (defalut " << icp_max_iter_ << " )" << std::endl;
	std::cout << "		--icp_corr_distance val	| ICP correspondence distance (defalut " << icp_corr_distance_ << " )" << std::endl;
	std::cout << "		--hv_clutter_reg val	| Clurrer Regularizer (defalut " << hv_clutter_reg_ << " )" << std::endl;
	std::cout << "		--hv_inlier_th val		| Inlier threshold (defalut " << hv_inlier_th_ << " )" << std::endl;
	std::cout << "		--hv_occlusion_th val	| Occlusion threshold (defalut " << hv_occlusion_th_ << " )" << std::endl;
	std::cout << "		--hv_rad_clutter val	| Clutter radius (defalut " << hv_rad_clutter_ << " )" << std::endl;
	std::cout << "		--hv_regularizer val	| Regularizer value (defalut " << hv_regularizer_ << " )" << std::endl;
	std::cout << "		--hv_rad_clutter val	| Normals radius (defalut " << hv_rad_normals_ << " )" << std::endl;
	std::cout << "		--hv_detect_clutter val | True if clustter detect enable (defalut " << hv_detect_clutter_ << " )" << std::endl;
	std::cout << "		" << std::endl;
	std::cout << "****************************************************************" << std::endl;
}

int main(int argc ,char** argv) {

	std::cout << "recognition global hypothesis verification." << std::endl;
	showHelp(argv[0]);


	return 0;
}