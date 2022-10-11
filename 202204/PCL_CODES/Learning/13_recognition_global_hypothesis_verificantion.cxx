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
#include <pcl/recognition/hv/hv_go.h>
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
		if (pcl::console::parse_argument(argc ,argv, "--alogrithm" ,used_algorithm) != -1) {
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

int main(int argc, char** argv) {

	std::cout << "recognition global hypothesis verification." << std::endl;
	parseCommandLine(argc, argv);

	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<NormalType>::Ptr model_normals(new pcl::PointCloud<NormalType>);
	pcl::PointCloud<NormalType>::Ptr scene_normals(new pcl::PointCloud<NormalType>);
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>);
	pcl::PointCloud<DescriptorType>::Ptr scene_descriptors(new pcl::PointCloud<DescriptorType>);

	if (pcl::io::loadPCDFile(model_filename_, *model) < 0) {
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
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descri_est;
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

	// ======================按KdTree存储方法匹配两个点云分组=======================
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences()); //最佳匹配组
	pcl::KdTreeFLANN<DescriptorType> match_search;
	match_search.setInputCloud(model_descriptors);
	std::vector<int> model_good_keypoints_indices;
	std::vector<int> scene_good_keypoints_indices;
	for (size_t i = 0; i < scene_descriptors->size(); i++) {
		std::vector<int> neigh_indices(1);
		std::vector<float> neigh_sqr_dists(1);
		if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) {
			continue;
		}
		// 寻找1个临近点
		int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
		if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) { // 临近点距离一般0-1之间
			pcl::Correspondence corr(neigh_indices[0], static_cast<int>(i), neigh_sqr_dists[0]);
			model_scene_corrs->push_back(corr);
			model_good_keypoints_indices.push_back(corr.index_query); //模型点云匹配 关键点键
			scene_good_keypoints_indices.push_back(corr.index_match); //场景点云匹配 关键点值
		}
	}
	pcl::PointCloud<PointType>::Ptr model_good_kp(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene_good_kp(new pcl::PointCloud<PointType>);
	pcl::copyPointCloud(*model_keypoints, model_good_keypoints_indices, *model_good_kp);
	pcl::copyPointCloud(*scene_keypoints, scene_good_keypoints_indices, *scene_good_kp);
	std::cout << "Match models points " << model_good_kp->size() << " scene_points " << scene_good_kp->size() << std::endl;

	// ======================实现匹配方式 执行聚类=======================
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
	std::vector<pcl::Correspondences> clustered_corrs; //匹配点相互连线
	// clustered_corrs[i][j].index_query
	// clustered_corrs[i][j].index_match
	std::cout << "Used Alogrithm : " << (used_hough_ == true ? "Hough " : "GC") << std::endl;
	if (used_hough_) {
		pcl::PointCloud<RFType>::Ptr model_rf(new pcl::PointCloud<RFType>);
		pcl::PointCloud<RFType>::Ptr scene_rf(new pcl::PointCloud<RFType>);
		pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
		rf_est.setFindHoles(true);
		rf_est.setRadiusSearch(rf_rad_);

		rf_est.setInputCloud(model_keypoints);
		rf_est.setInputNormals(model_normals);
		rf_est.setSearchSurface(model);
		rf_est.compute(*model_rf);

		rf_est.setInputCloud(scene_keypoints);
		rf_est.setInputNormals(scene_normals);
		rf_est.setSearchSurface(scene);
		rf_est.compute(*scene_rf);

		// 执行聚类Clustering
		pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
		clusterer.setHoughBinSize(cg_size_);
		clusterer.setHoughThreshold(cg_thresh_);
		clusterer.setUseInterpolation(true);
		clusterer.setUseDistanceWeight(false);

		clusterer.setInputCloud(model_keypoints);
		clusterer.setInputRf(model_rf);
		clusterer.setSceneCloud(scene_keypoints);
		clusterer.setSceneRf(scene_rf);
		clusterer.setModelSceneCorrespondences(model_scene_corrs);

		clusterer.recognize(rototranslations, clustered_corrs);
	}
	else {
		pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
		gc_clusterer.setGCSize(cg_size_);
		gc_clusterer.setGCThreshold(cg_thresh_);
		gc_clusterer.setInputCloud(model_keypoints);
		gc_clusterer.setSceneCloud(scene_keypoints);
		gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);
		gc_clusterer.recognize(rototranslations, clustered_corrs);
	}

	if (rototranslations.size() <= 0) {
		std::cout << "Recognized not found instance." << std::endl;
		return -3;
	}
	else {
		std::cout << "Recognized Instances " << rototranslations.size() << std::endl;
	}
	/***
		Generate clouds for each instances found
	**/
	std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;
	for (size_t i = 0; i < rototranslations.size(); i++) {
		pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>);
		pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);
		instances.push_back(rotated_model);
	}

	// ======================ICP点云配准=======================
	std::vector<pcl::PointCloud<PointType>::ConstPtr> registered_instances;
	if (true) {
		std::cout << "--------------- ICP -----------------------" << std::endl;
		for (size_t i = 0; i < rototranslations.size(); i++) {
			pcl::IterativeClosestPoint<PointType, PointType> icp;
			icp.setMaximumIterations(icp_max_iter_);
			icp.setMaxCorrespondenceDistance(icp_corr_distance_);
			icp.setInputTarget(scene); //场景
			icp.setInputSource(instances[i]); //场景中的实例
			pcl::PointCloud<PointType>::Ptr registered(new pcl::PointCloud<PointType>);
			icp.align(*registered); //匹配到的点云
			registered_instances.push_back(registered);
			std::cout << "Instance " << i << " ";
			if (icp.hasConverged()) {
				std::cout << "Aligned!" << std::endl;
			}
			else {
				cout << "Not Aligned!" << std::endl;
			}
		}
		std::cout << "Registered points " << registered_instances.size() << std::endl;
		std::cout << "-------------------------------------------" << std::endl;
	}

	// ======================模型假设性检验 Global hypotheses verfication =======================
	std::vector<bool> hypotheses_mask; // Mask Vector to identify positive hypotheses
	pcl::GlobalHypothesesVerification<PointType, PointType> GoHv;
	GoHv.setSceneCloud(scene);
	GoHv.addModels(registered_instances, true); // Model to verify

	GoHv.setInlierThreshold(hv_inlier_th_);
	GoHv.setOcclusionThreshold(hv_occlusion_th_);
	GoHv.setRegularizer(hv_regularizer_);
	GoHv.setRadiusClutter(hv_rad_clutter_);
	GoHv.setClutterRegularizer(hv_clutter_reg_);
	GoHv.setDetectClutter(hv_detect_clutter_);
	GoHv.setRadiusNormals(hv_rad_normals_);
	GoHv.verify();
	GoHv.getMask(hypotheses_mask); //i-element True if hvModel[i] verfies hypotheses

	std::cout << "---------- Hypotheses Verification ---------" << std::endl;
	for (size_t i = 0; i < hypotheses_mask.size(); i++) {
		if (hypotheses_mask[i]) {
			std::cout << "Instance " << i << " is GOOD." << std::endl;
		}
		else {
			std::cout << "Instance " << i << " is BAD." << std::endl;
		}
	}
	std::cout << "-------------------------------------------" << std::endl;

	// ====================== 可视化 =======================
	pcl::visualization::PCLVisualizer viewer("Hypotheses Verification");
	viewer.addPointCloud(scene, "scene_cloud");
	pcl::PointCloud<PointType>::Ptr off_scene_model(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr off_model_good_kp(new pcl::PointCloud<PointType>);

	pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*model_good_kp, *off_model_good_kp, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	if (show_keypoints_) {
		CloudStyle modelStyle = style_white;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler(off_scene_model, modelStyle.r, modelStyle.g, modelStyle.b);
		viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, modelStyle.size, "off_scene_model");
	}
	if (show_keypoints_) {
		CloudStyle goodKeypointStyle = style_violet;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> model_good_keypoints_color_handler(off_model_good_kp, goodKeypointStyle.r, goodKeypointStyle.g, goodKeypointStyle.b);
		viewer.addPointCloud(off_model_good_kp, model_good_keypoints_color_handler, "model_good_keypoints");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "model_good_keypoints");

		pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_good_keypoints_color_handler(scene_good_kp, goodKeypointStyle.r, goodKeypointStyle.g, goodKeypointStyle.b);
		viewer.addPointCloud(scene_good_kp, scene_good_keypoints_color_handler, "scene_good_keypoints");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, goodKeypointStyle.size, "scene_good_keypoints");
	}
	for (size_t i = 0; i < instances.size(); i++) {
		std::stringstream ss_instance;
		ss_instance << "instance_" << i;

		CloudStyle clusterStyle = style_red;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> instance_color_handler(instances[i] ,clusterStyle.r ,clusterStyle.g ,clusterStyle.b);
		viewer.addPointCloud(instances[i] ,instance_color_handler ,ss_instance.str());
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,clusterStyle.size ,ss_instance.str());

		CloudStyle registeredStyles = hypotheses_mask[i] ? style_green : style_cyan;
		ss_instance << "_registered" << std::endl;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> registered_instance_color_handler(registered_instances[i] ,registeredStyles.r , registeredStyles.g , registeredStyles.b);
		viewer.addPointCloud(registered_instances[i] , registered_instance_color_handler ,ss_instance.str());
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,registeredStyles.size ,ss_instance.str());
	}

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}
	return 0;
}
