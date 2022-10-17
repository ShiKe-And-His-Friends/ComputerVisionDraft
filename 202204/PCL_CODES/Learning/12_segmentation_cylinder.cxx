/***

	分割  圆柱模型分割，使用随机一致性采样

**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace Segmentation_Cylinder {

	typedef pcl::PointXYZ PointT;

};

using namespace Segmentation_Cylinder;

int segmentation_cylinder(int argc ,char **argv) {

	std::cout << "Segmentation cylinder contours." << std::endl;

	// intilize values
	pcl::PCDReader reader;
	pcl::PassThrough<PointT> pass;
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
	pcl::PCDWriter writer;
	pcl::ExtractIndices<PointT> extract;
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

	// Datasets


	return 0;
}
