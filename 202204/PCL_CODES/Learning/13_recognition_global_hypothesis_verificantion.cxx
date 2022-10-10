/***
	
	识别 3D物体假设识别

	Global Hypothesis Verification  全局性假设

**/
#include <iostream>
#include <Eigen>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_based_signature.h>
#include <pcl/features/normal_3d_omp.h>
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

namespace Recoginition_Global_Hythcesis_Verificaiton {

	typedef pcl::PointXYZRGBA PointType;
	typedef pcl::Normal NormalType;
	typedef pcl::ReferenceFrame RFType; //参考帧
	typedef pcl::SHOT352 DescriptorType; // SHOT特征 (32 * 11 = 352)

	// 点云风格  颜色 大小 结构体
	struct CloudStyle {

		double r;
		double g;
		double b;
		double size;

		CloudStyle(double r, double g, double b, double size) :
			r(r),
			g(g),
			b(b),
			size(size) {

		}
	};


	CloudStyle style_white(255.0, 255.0, 255.0, 4.0);
	CloudStyle style_red(255.0, 0, 0, 3.0);
	CloudStyle style_green(0, 255.0, 0.0, 5.0);
	CloudStyle style_cyan(93.0, 0.0, 217.0, 4.0);
	CloudStyle style_violet(255.0, 0.0, 255.0, 8.0);

	std::string model_filename_;
	std::string scene_filename_;

	// 算法参数 Algorithm params
	bool show_keypoints_(false);
	bool used_hough_(true);
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


	void showHelp(char* filename) {
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
		std::cout << "		--model_ss val			| Model uniform sampling radius (defalut " << model_ss_ << " )" << std::endl;
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

	void parseCommandLine(int argc, char* argv[]) {
		if (pcl::console::find_switch(argc, argv, "-h")) {
			showHelp(argv[0]);
			exit(0);
		}
		std::vector<int> filenames;
		filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
		if (filenames.size() != 2) {
			std::cout << "Filenames missing." << std::endl;
			showHelp(argv[0]);
			exit(-1);
		}

		model_filename_ = argv[filenames[0]];
		scene_filename_ = argv[filenames[1]];

		if (pcl::console::find_switch(argc, argv, "-k")) {
			show_keypoints_ = true;
		}
		std::string used_algorithm;
		if (pcl::console::parse_argument(argc ,argv, "--algorithm" ,used_algorithm) != -1) {
			if (used_algorithm.compare("Hough") == 0) {
				used_hough_ = true;
			}
			else if (used_algorithm.compare("GC") == 0) {
				used_hough_ = false;
			}
			else {
				std::cout << "Wrong algorithm name." << std::endl;
				showHelp(argv[0]);
				exit(-2);
			}
		}

		// General parameters
		pcl::console::parse_argument(argc ,argv ,"--model_ss",model_ss_);
		pcl::console::parse_argument(argc, argv, "--scene_ss", scene_ss_);
		pcl::console::parse_argument(argc, argv, "--rf_rad", rf_rad_);
		pcl::console::parse_argument(argc, argv, "--descr_rad", descr_rad_);
		pcl::console::parse_argument(argc, argv, "--cg_size", cg_size_);
		pcl::console::parse_argument(argc, argv, "--icp_max_iter",icp_max_iter_);
		pcl::console::parse_argument(argc, argv, "--icp_corr_distance", icp_corr_distance_);
		pcl::console::parse_argument(argc, argv, "--hv_clutter_reg", hv_clutter_reg_);
		pcl::console::parse_argument(argc, argv, "--hv_inlier_th", hv_inlier_th_);
		pcl::console::parse_argument(argc, argv, "--hv_occlusion_th", hv_occlusion_th_);
		pcl::console::parse_argument(argc, argv, "--hv_rad_cluter", hv_rad_clutter_);
		pcl::console::parse_argument(argc, argv, "--hv_regularizer", hv_regularizer_);
		pcl::console::parse_argument(argc, argv, "--hv_rad_normals", hv_rad_normals_);
		pcl::console::parse_argument(argc, argv, "--hv_detect_clutter", hv_detect_clutter_);
	}

}

using namespace Recoginition_Global_Hythcesis_Verificaiton;

int main(int argc ,char** argv) {

	std::cout << "recognition global hypothesis verification." << std::endl;
	parseCommandLine(argc ,argv);

	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<NormalType>::Ptr model_normals(new pcl::PointCloud<NormalType>);
	pcl::PointCloud<NormalType>::Ptr scene_normals(new pcl::PointCloud<NormalType>);
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>);
	pcl::PointCloud<DescriptorType>::Ptr scene_descriptors(new pcl::PointCloud<DescriptorType>);

	if (pcl::io::loadPCDFile(model_filename_ ,*model) <0) {
		std::cout << "Load model file error." << std::endl;
		showHelp(argv[0]);
		return -1;
	}
	if (pcl::io::loadPCDFile(scene_filename_, *scene) < 0) {
		std::cout << "Load scene file error." << std::endl;
		showHelp(argv[0]);
		return -2;
	}

	// ======================计算法向量 曲率=======================
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est; //法线
	norm_est.setKSearch(10); //最近10个点，协方差矩阵PCA分解
	norm_est.setInputCloud(model);
	norm_est.compute(*model_normals);
	norm_est.setInputCloud(scene);
	norm_est.compute(*scene_normals);

	// ======================下采样获取关键点=======================
	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud(model);
	uniform_sampling.setRadiusSearch(model_ss_);
	uniform_sampling.filter(*model_keypoints);
	std::cout << "Model total keypoints " << model_keypoints->size() << std::endl;
	uniform_sampling.setInputCloud(scene);
	uniform_sampling.setRadiusSearch(scene_ss_);
	uniform_sampling.filter(*scene_keypoints);
	std::cout << "Scene total keypoints " << scene_keypoints->size() << std::endl;

	// ======================SHOT描述子=======================
	pcl::SHOTColorEstimationOMP<PointType, NormalType, DescriptorType> descri_est;
	descri_est.setRadiusSearch(descr_rad_);
	descri_est.setInputCloud(model_keypoints);
	descri_est.setInputNormals(model_normals);
	descri_est.setSearchSurface(model);
	descri_est.compute(*model_descriptors);
	std::cout << "Model total descriptors " << model_descriptors->size() << std::endl;

	descri_est.setInputCloud(scene_keypoints);
	descri_est.setInputNormals(scene_normals);
	descri_est.setSearchSurface(scene);
	descri_est.compute(*scene_descriptors);
	std::cout << "Scene total descriptors " << scene_descriptors->size() << std::endl;

	return 0;
}