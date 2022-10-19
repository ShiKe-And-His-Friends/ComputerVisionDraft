/***

	Surface  hull

	平面模型提取凸包

**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>

int surface_hull_2d(int argc ,char **argv) {

	std::cout << "Surface hull 2d" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PCDReader reader;

	reader.read("table_scene_mug_stereo_textured.pcd",*cloud);
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0 ,1.1);
	pass.filter(*cloud_filtered);
	std::cerr << "PointCloud after filtering has:" <<
		cloud_filtered->points.size() << " data points."
		<< std::endl;
	// segmentation
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.01);

	seg.setInputCloud(cloud_filtered);
	seg.segment(*inliers ,*coefficients);
	std::cerr << "Point cloud after segmentation has : " << inliers->indices.size() << " inliers." << std::endl;

	// project the model inliers
	pcl::ProjectInliers<pcl::PointXYZ> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setIndices(inliers);
	proj.setInputCloud(cloud_filtered);
	proj.setModelCoefficients(coefficients);
	proj.filter(*cloud_projected);
	std::cerr << "Point cloud after projection has :" << cloud_projected->points.size() << std::endl;

	// save
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ConcaveHull<pcl::PointXYZ> chull;
	chull.setInputCloud(cloud_projected);
	chull.setAlpha(0.1);
	chull.reconstruct(*cloud_hull);

	std::cerr << "Concave hull has : " << cloud_hull->points.size() <<
		" data points." << std::endl;
	pcl::PCDWriter writer;
	writer.write("table_scene_mug_stereo_textured_hull2.pcd" ,*cloud_hull ,false);

	return 0;
}