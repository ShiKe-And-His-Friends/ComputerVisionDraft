/****

	匹配 刚性物体匹配性检测
	     鲁棒位姿标记

	alignment prerejective

**/
#include <Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

int registration_align_prerejection(int argc ,char **argv) {

	std::cout << "registration alignment prerejective." << std::endl;

	PointCloudT::Ptr object(new PointCloudT);
	PointCloudT::Ptr object_aligned(new PointCloudT);
	PointCloudT::Ptr scene(new PointCloudT);
	FeatureCloudT::Ptr object_features(new FeatureCloudT);
	FeatureCloudT::Ptr scene_features(new FeatureCloudT);

	if (argc != 3) {
		pcl::console::print_error("Syntax is : %s object.pcd scene.pcd \n" ,argv[0]);
		return -1;
	}
	pcl::console::print_highlight("Loading point clouds ...\n");
	if (pcl::io::loadPCDFile<PointNT>(argv[1] ,*object) < 0 || pcl::io::loadPCDFile<PointNT>(argv[2] ,*scene) < 0) {
		pcl::console::print_error("Error loading object/scene file! \n");
		return -2;
	}
	// 1.为加快处理速度，使用0.005提速分辨率进行下采样
	pcl::console::print_highlight("Downsampling ...\n");
	pcl::VoxelGrid<PointNT> grid;
	const float leaf = 0.005f;
	grid.setLeafSize(leaf ,leaf ,leaf);
	grid.setInputCloud(object);
	grid.filter(*object);
	grid.setInputCloud(scene);
	grid.filter(*scene);

	// 2.估计场景法线
	pcl::console::print_highlight("Estimating scene normals....\n");
	pcl::NormalEstimationOMP<PointNT, PointNT> nest;
	nest.setRadiusSearch(0.01);
	nest.setInputCloud(scene);
	nest.compute(*scene);

	// 3.特征估计
	// 对下采样的每个点云，我们使用FPFH快速直方描述子来描述
	pcl::console::print_highlight("Estimating features...\n");
	FeatureEstimationT fest;
	fest.setRadiusSearch(0.025);
	fest.setInputCloud(object);
	fest.setInputNormals(object);
	fest.compute(*object_features);
	fest.setInputCloud(scene);
	fest.setInputNormals(scene);
	fest.compute(*scene_features);

	// 4.对齐配准对象创建与配置，实施配准Perform alignment
	pcl::console::print_highlight("Starting alignment...\n");
	pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align; //基于采样一致性的位姿估计
	align.setInputSource(object);
	align.setSourceFeatures(object_features);
	align.setInputTarget(scene);
	align.setTargetFeatures(scene_features);
	align.setMaximumIterations(50000);
	align.setNumberOfSamples(3); //场景之间进行采样点的对应个数,至少需要3个点
	align.setCorrespondenceRandomness(5);
	align.setSimilarityThreshold(0.9f);
	align.setMaxCorrespondenceDistance(2.5f * leaf);
	align.setInlierFraction(0.25f);

	{
		pcl::ScopeTime t("Alignment");
		align.align(*object_aligned); //对齐的对象存储在点云object_aligned中
	}
	if (align.hasConverged()) {
		printf("\n");
		Eigen::Matrix4f tranformation = align.getFinalTransformation();
		std::cout << "Rotation matrix:" << std::endl;
		printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", tranformation(0, 0), tranformation(0, 1), tranformation(0, 2));
		printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", tranformation(1, 0), tranformation(1, 1), tranformation(1, 2));
		printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", tranformation(2, 0), tranformation(2, 1), tranformation(2, 2));
		std::cout << "Translation vector:" << std::endl;
		printf("\tt=<%6.3f\t%6.3f\t%6.3f\t>\n", tranformation(0, 3), tranformation(1, 3), tranformation(2, 3));

		pcl::visualization::PCLVisualizer visu("Alignment-刚性物体的鲁棒位姿估计");
		visu.addPointCloud(scene ,ColorHandlerT(scene ,0.0 ,255.0 ,0.0), "scene");
		visu.addPointCloud(object_aligned ,ColorHandlerT(object_aligned ,0.0 ,0.0 ,255.0),"object_aligned");
		visu.spin();
	}
	else {
		pcl::console::print_error("Alignment failed!\n");
		return -3;
	}

	return 0;
}
