/***

	识别  基于对应分组的三维物体识别
	[1] 计算法线向量  近邻邻域内 协方差矩阵PCA  降维到二维平面 计算法线向量
		PCA降维原理：
			对一堆杂乱无序的三维点云，这样计算法线，
			（1） 对每个点，取邻近点，比如50个邻近点，会用到K-D树
			（2） 对临近点做PCA降维，降维到二维平面上，这个平面必然是它的切平面
			（3） 切平面的法线就是该点的法线，法线的正负方向选择，需要判断凸包
	[2] 下采样滤波使用均匀采样得到关键点（或者尝试体素格子下采样）
	[3] 为Keypoints关键点计算SHOT描述子
	[4] 按存储方法KDTree匹配两个点云（描述子向量匹配），点云分组得到匹配的组，描述点的对应关系
	[5] 参考霍夫聚类/集合一致性聚类，匹配点云cluster，平移矩阵和匹配点关系
	[6] 分组显示 平移矩阵T 将模型点云T变换后显示 以及显示 点对点之间的关系

**/
#include <iostream>
#include <Eigen>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h> //分组算法对应两个实体的匹配（描述符等）
#include <pcl/features/normal_3d_omp.h> //法向量特征
#include <pcl/features/shot_omp.h> //SHOT描述子
#include <pcl/features/board.h> 
#include <pcl/filters/uniform_sampling.h> //均匀采样滤波
#include <pcl/recognition/cg/hough_3d.h> // Hough算子
#include <pcl/recognition/cg/geometric_consistency.h> // 几何一致性
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h> //kdtree快速近邻搜索
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

/*
	SHOT 特征子
	查询p点为中心半径r的球形区域，沿着径向2次、方位8次、俯仰2次划分网络为32个区域
	每个空间区域计算区域点的法线nv和中心点p法线np之间的夹角余弦cosX = nv*np
	再根据计算的余弦值对落入空间区域的点数进行直方图统计(划分11个)
	对计算结果归一化，使得点云密度具有鲁棒性，得到352特征(32*11 = 352)
	Atircle: Unique Signatures of Histograms for Local Surface
*/

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;//参考帧
typedef pcl::SHOT352 DescriptorType; //SHOT描述子 32 * 11 = 352

std::string model_filename;
std::string scene_filename;

// 算法参数
bool show_keypoints_(false);
bool show_correspondences_(false);
bool use_cloud_resolution_(false);
bool use_hough_(true);
float model_ss_(0.01f); //模型采样率
float scene_ss_(0.03f); //场景采样率
float rf_rad_(0.015f);
float descr_rad_(0.02f);
float cg_size_(0.01f); //聚类，霍夫空间设置每个bin的大小
float cg_thresh_(5.0f); //聚类阈值

void showHelp(char *filename) {
	std::cout << std::endl;
	std::cout << "********************************************************" << std::endl;
	std::cout << "*		" << std::endl;
	std::cout << "*		Correspondence Grouping Tutorial - Usage Guide" << std::endl;
	std::cout << "*		" << std::endl;
	std::cout << "********************************************************" << std::endl;
	std::cout << "Usage: " <<filename << " model_filename.pcd scene_filename.pcd [options]" << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << "		-h Show this help" << std::endl;
	std::cout << "		-k Show used keypoints" << std::endl;
	std::cout << "		-c Show used correspondence," << std::endl; //分组算法
	std::cout << "		-r Compute the model cloud resolution and multiply" << std::endl;
	std::cout << "		" << std::endl;
	std::cout << "		--algorithm (Hough|GC) Clustering alogrithm used (default Hough)" << std::endl;
	std::cout << "		--model_ss val Model uniform sampling radius (default 0.01)" << std::endl;
	std::cout << "		--scene_ss val Scene uniform sampling radius (default 0.03)" << std::endl;
	std::cout << "		--rf_rad_val Reference frame radius (default 0.015)" << std::endl;
	std::cout << "		--descr_rad val Desciptor radius (default 0.02)" << std::endl;
	std::cout << "		--cg_size val Cluster size (default 0.01)" << std::endl;
	std::cout << "		--cg_thresh val Clustering threshold (default 5)" << std::endl;
}

void parseCommandLine(int argc,char *argv[]) {
	// milk.pcd milk_cartoon_all_small_clorox.pcd -k -c -r --algorithm "Hough" --model_ss 0.01f  --scene_ss 0.03f  --rf_rad 0.015f  --descr_rad 0.02f --cg_size 0.01f --cg_thresh 5.0f
	if (pcl::console::find_switch(argc, argv ,"-h")) {
		showHelp(argv[0]);
		exit(0);
	}
	// model file & scene file
	std::vector<int> filenames;
	filenames = pcl::console::parse_file_extension_argument(argc ,argv ,".pcd");
	if (filenames.size() != 2) {
		std::cout << "Filenames not missing." << std::endl;
		showHelp(argv[0]);
		exit(-1);
	}
	model_filename = argv[filenames[0]];
	scene_filename = argv[filenames[1]];
	std::cout << "model file : " << model_filename << std::endl;
	std::cout << "scene file : " << scene_filename << std::endl;
	if (pcl::console::find_switch(argc,argv,"-k")) {
		show_keypoints_ = true;
	}
	if (pcl::console::find_switch(argc ,argv ,"-c")) {
		show_correspondences_ = true; //显示对应分组
	}
	if (pcl::console::find_switch(argc ,argv ,"-r")) {
		use_cloud_resolution_ = true;
	}
	//聚类算法
	std::string used_algorithm;
	if (pcl::console::parse_argument(argc ,argv,"--algorithm" ,used_algorithm) != -1) {
		if (used_algorithm.compare("Hough") == 0) {
			use_hough_ = true;
		}
		else if (used_algorithm.compare("GC") == 0) {
			use_hough_ = false;
		}
		else {
			std::cout << "Wrong algorithm name." << std::endl;
			showHelp(argv[0]);
			exit(-2);
		}
	}
	pcl::console::parse_argument(argc,argv ,"--model_ss" ,model_ss_);
	pcl::console::parse_argument(argc,argv,"--scene_ss" ,scene_ss_);
	pcl::console::parse_argument(argc ,argv,"--rf_rad" ,rf_rad_);
	pcl::console::parse_argument(argc ,argv, "--descr_rad" ,descr_rad_);
	pcl::console::parse_argument(argc ,argv ,"--cg_size" ,cg_size_);
	pcl::console::parse_argument(argc ,argv ,"--cg_thresh" ,cg_thresh_);
}

double computeCloudResolution(const pcl::PointCloud<PointType>::ConstPtr &cloud) {
	double res = 0.0;
	int n_points = 0;
	int nres; // 临近点索引
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2); //距离平方
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);
	
	for (size_t i = 0; i < cloud->size(); i++) {
		if (!std::isfinite((*cloud)[i].x)) {
			continue;
		}
		nres = tree.nearestKSearch(i ,2 ,indices ,sqr_distances);
		if (nres == 2) { // 最近第一个为自身，第二个为离自己最近的点
			res += sqrt(sqr_distances[1]); // 开根号后相加
			++n_points;
		}
	}
	if (n_points != 0) {
		res /= n_points;
	}
	return res;
}

int main(int argc ,char **argv) {

	std::cout << "registition correspondence grouping..." << std::endl;
	parseCommandLine(argc ,argv);

	// ======================载入点云=======================
	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr model_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr scene_keypoints(new pcl::PointCloud<PointType>);
	pcl::PointCloud<NormalType>::Ptr model_normals(new pcl::PointCloud<NormalType>);
	pcl::PointCloud<NormalType>::Ptr scene_normals(new pcl::PointCloud<NormalType>);
	pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>);
	pcl::PointCloud<DescriptorType>::Ptr scene_descriptors(new pcl::PointCloud<DescriptorType>);
	
	if (pcl::io::loadPCDFile(model_filename ,*model) < 0) {
		std::cout << "Error loading model cloud." << std::endl;
		showHelp(argv[0]);
		return -1;
	}
	if (pcl::io::loadPCDFile(scene_filename, *scene) < 0) {
		std::cout << "Error loading scene cloud." << std::endl;
		showHelp(argv[0]);
		return -1;
	}

	// ================设置分辨率变量========================
	if (use_cloud_resolution_) {
		float resolution = static_cast<float>(computeCloudResolution(model));
		if (resolution != 0.0f) {
			model_ss_ *= resolution;
			scene_ss_ *= resolution;
			rf_rad_ *= resolution;
			descr_rad_ *= resolution;
			cg_size_ *= resolution;
		}
		std::cout << "Model resolution: " << resolution << std::endl;
		std::cout << "Model sampling size: " << model_ss_ << std::endl;
		std::cout << "Scene sampling size: " << scene_ss_ << std::endl;
		std::cout << "LRF support radius: " << rf_rad_ << std::endl;
		std::cout << "SHOT descripotr radius: " << descr_rad_ << std::endl;
		std::cout << "Clustering bin size: " << cg_size_ << std::endl;
	}

	// ==================计算法向量=======================
	pcl::NormalEstimationOMP<PointType, NormalType> norm_est; //  多核 计算法线模型 OpenMP
	norm_est.setKSearch(10); //最近10个点 协方差矩阵PCA分解
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	//norm_est.setSearchMethod(tree); //多核模式不需要设置
	norm_est.setInputCloud(model);
	norm_est.compute(*model_normals);
	norm_est.setInputCloud(scene);
	norm_est.compute(*scene_normals);

	// ==========下采样滤波使用均匀采样得到关键点（或者尝试体素格子下采样）============
	pcl::UniformSampling<PointType> uniform_sampling;
	uniform_sampling.setInputCloud(model);
	uniform_sampling.setRadiusSearch(model_ss_);
	uniform_sampling.filter(*model_keypoints);
	std::cout << "Model total points:" << model->size() << " ;Selected Keypoints: " << model_keypoints->size() << std::endl;
	uniform_sampling.setInputCloud(scene);
	uniform_sampling.setRadiusSearch(scene_ss_);
	uniform_sampling.filter(*scene_keypoints);
	std::cout << "Scene total points:" << scene->size() << " ;Selected Keypoints: " << scene_keypoints->size() << std::endl;

	// ==========为keypoints关键点计算添加描述子Hough / SHOT==============
	pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est; //SHOT描述子
	descr_est.setRadiusSearch(descr_rad_); // 调大描述子半径 修复 ‘The local reference frame is not valid!’
	descr_est.setInputCloud(model_keypoints);
	descr_est.setInputNormals(model_normals);
	descr_est.setSearchSurface(model);
	descr_est.compute(*model_descriptors); //模型点云描述子

	descr_est.setInputCloud(scene_keypoints);
	descr_est.setInputNormals(scene_normals);
	descr_est.setSearchSurface(scene);
	descr_est.compute(*scene_descriptors);

	std::cout << "Model descriptors points:" << model_descriptors->size() << std::endl;
	std::cout << "Scene descriptors points:" << scene_descriptors->size() << std::endl;
	// =================按照KdTree匹配两个点云，点云分组得到匹配的组===================
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences()); //最佳匹配分组
	pcl::KdTreeFLANN<DescriptorType> match_search;
	match_search.setInputCloud(model_descriptors);
	for (size_t i = 0; i < scene_descriptors->size();i++) {
		std::vector<int> neigh_indices(1); // 索引
		std::vector<float> neigh_sqr_dists(1);
		if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) { // 跳过NAN点
			continue;
		}
		int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i) ,1 ,neigh_indices ,neigh_sqr_dists);
		if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) {
			pcl::Correspondence corr(neigh_indices[0] ,static_cast<int>(i) ,neigh_sqr_dists[0]);
			// 模型点云 和 场景点云 的最佳匹配,距离neigh_sqr_dists[0]
			model_scene_corrs->push_back(corr);
		}
	}
	std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

	// =============== 执行聚类 ===================
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations; // 变换矩阵和旋转聚合矩阵的集合
	// [!!!]eigen中固定大小的类使用STL容器的时候，如果直接使用会出错，需要使用Eigen::aligened_allocator对齐
	std::vector<pcl::Correspondences> clustered_corrs; //匹配点 相互连线的索引
	// clustered_corrs[i][j].index_query 模型点 索引
	// clustered_corrs[i][j].index_match 场景点 索引
	
	//使用Hough3D 霍夫算法 寻找匹配点
	if (use_hough_) {
		// 计算参考帧的Hough，也就是关键点
		pcl::PointCloud<RFType>::Ptr model_rf(new pcl::PointCloud<RFType>());
		pcl::PointCloud<RFType>::Ptr scene_rf(new pcl::PointCloud<RFType>());
		// 估计模型参考帧的特征 点云 法线 参考帧
		pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
		rf_est.setFindHoles(true);
		rf_est.setRadiusSearch(rf_rad_);
		rf_est.setInputCloud(model_keypoints);
		rf_est.setInputNormals(model_normals);
		rf_est.setSearchSurface(model);
		rf_est.compute(*model_rf);

		rf_est.setInputCloud(scene);
		rf_est.setInputNormals(scene_normals);
		rf_est.setSearchSurface(scene);
		rf_est.compute(*scene_rf);

		// 对输入点与的聚类，区分不同的实例场景
		pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
		clusterer.setHoughBinSize(cg_size_); //霍夫空间每个bin大小
		clusterer.setHoughThreshold(cg_thresh_);
		clusterer.setUseInterpolation(true);
		clusterer.setUseDistanceWeight(false);

		clusterer.setInputCloud(model_keypoints);
		clusterer.setInputRf(model_rf); //模型点云参考帧
		clusterer.setSceneCloud(scene_keypoints);
		clusterer.setSceneRf(scene_rf); //场景点云参考帧
		clusterer.setModelSceneCorrespondences(model_scene_corrs); // 组关系
		//clusterer.cluster(clustered_corrs);
		clusterer.recognize(rototranslations ,clustered_corrs);

	}
	else { // CG 几何一致性 寻找匹配点
		pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
		gc_clusterer.setGCSize(cg_size_);
		gc_clusterer.setGCThreshold(cg_thresh_);
		gc_clusterer.setInputCloud(model_keypoints);
		gc_clusterer.setSceneCloud(scene_keypoints);
		gc_clusterer.setModelSceneCorrespondences(model_scene_corrs);
		//gc_clusterer.cluster(clustered_corrs);
		gc_clusterer.recognize(rototranslations ,clustered_corrs);
	}

	// =============== 输出识别结果 ===================
	std::cout << "Model instances found: " << rototranslations.size() << std::endl;
	for (size_t i = 0; i < rototranslations.size();i++) {
		std::cout << "Instance " << i+1 << std::endl;
		std::cout << "  Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

		// 打印 [R T]
		Eigen::Matrix3f rotation = rototranslations[i].block<3, 3 >(0,0);
		Eigen::Matrix3f translation = rototranslations[i].block<3, 1>(0 ,3);

		std::cout << "Rotation matrix:" << std::endl;
		printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
		printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
		printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", rotation(2, 0), rotation(2, 1), rotation(2, 2));

		std::cout << "Translation vector:" << std::endl;
		printf("\tt=<%6.3f\t%6.3f\t%6.3f\t>\n", translation(0, 3), translation(1, 3), translation(2, 3));
	}

	// =============== 可视化 ===================
	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.addPointCloud(scene ,"scene_cloud");

	pcl::PointCloud<PointType>::Ptr off_scene_model(new pcl::PointCloud<PointType>); //变换后的模型点云
	pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints(new pcl::PointCloud<PointType>); //关键点
	if (show_correspondences_ || show_keypoints_) { // 可视化平移后的模型点云
		// 对输入点云进行旋转和平移，在可视化界面中间x轴负方向平移一个单位
		pcl::transformPointCloud(*model ,*off_scene_model ,Eigen::Vector3f(-1 ,0 ,0) ,Eigen::Quaternionf(1,0,0,0));
		pcl::transformPointCloud(*model_keypoints ,*off_scene_model_keypoints ,Eigen::Vector3f(-1 ,0,0), Eigen::Quaternionf(1, 0, 0, 0));

		pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler(off_scene_model,255,255,128);
		viewer.addPointCloud(off_scene_model ,off_scene_model_color_handler ,"off_scene_model");
	}
	if (show_keypoints_) { //可视化 场景关键点和模型关键点
		pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoint_color_handler(scene_keypoints ,0,0,255);
		viewer.addPointCloud(scene_keypoints ,scene_keypoint_color_handler ,"scene_keypoints");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,5 ,"scene_keypoints");

		pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler(off_scene_model_keypoints ,0 ,0 ,255);
		viewer.addPointCloud(off_scene_model_keypoints,off_scene_model_keypoints_color_handler ,"off_scene_model_keypoints");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,5 ,"off_scene_model_keypoints");
	}
	for (size_t i = 0; i < rototranslations.size();i++) {//在场景中匹配点云
		pcl::PointCloud<PointType>::Ptr rotated_model(new pcl::PointCloud<PointType>);
		pcl::transformPointCloud(*model ,*rotated_model ,rototranslations[i]);
		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;
		pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler(rotated_model ,255 ,0 ,0);
		viewer.addPointCloud(rotated_model ,rotated_model_color_handler ,ss_cloud.str());
		if (show_correspondences_) {
			for (size_t j = 0; j < clustered_corrs[i].size(); j++) {
				std::stringstream ss_line;
				ss_line << i << "_" << j;
				PointType& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
				PointType& scene_point = scene_keypoints->at(clustered_corrs[i][j].index_match);

				viewer.addLine<PointType, PointType>(model_point, scene_point, 0, 255, 0, ss_line.str());
			}
		}
	}
	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;
}