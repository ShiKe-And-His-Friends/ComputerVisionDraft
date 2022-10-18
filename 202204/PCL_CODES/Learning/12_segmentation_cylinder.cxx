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
#include <boost/thread.hpp>

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
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
	pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients) ,coefficients_cylinder(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);

	// read in cloud data
	reader.read("table_scene_mug_stereo_textured.pcd" ,*cloud);
	std::cerr << "Point Cloud has: " << cloud->points.size() << " data points." << std::endl;

	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0 ,1.5);
	pass.filter(*cloud_filtered);
	std::cerr << "Point Cloud after filterning has: " << cloud_filtered->points.size() << std::endl;

	// normal line assume
	ne.setSearchMethod(tree);
	ne.setInputCloud(cloud_filtered);
	ne.setKSearch(50);
	ne.compute(*cloud_normals);

	// segemention object by planner
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
	seg.setNormalDistanceWeight(0.1);
	seg.setModelType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.03);
	seg.setInputCloud(cloud_filtered);
	seg.setInputNormals(cloud_normals);
	seg.segment(*inliers_plane ,*coefficients_plane);
	std::cerr << "Plane coefficients " << *coefficients_plane << std::endl;

	// extract point in plane
	extract.setInputCloud(cloud_filtered);
	extract.setIndices(inliers_plane);
	extract.setNegative(false);
	pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>);
	extract.filter(*cloud_plane);
	std::cerr<< "Point cloud representing the planar component: " << cloud_plane->points.size() << std::endl;
	writer.write("table_scene_mug_stereo_textured_plane.pcd" ,*cloud_plane ,false);
	// remove planar inliers ,extract the rest
	extract.setNegative(true);
	extract.filter(*cloud_filtered2);
	extract_normals.setNegative(true);
	extract_normals.setInputCloud(cloud_normals);
	extract_normals.setIndices(inliers_plane);
	extract_normals.filter(*cloud_normals);

	// create segement object for cylinder segementation and set all the parameter
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_CYLINDER);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setNormalDistanceWeight(0.1); //表面法线系数
	seg.setMaxIterations(10000);
	seg.setDistanceThreshold(0.05);
	seg.setRadiusLimits(0 ,0.1);
	seg.setInputCloud(cloud_filtered2);
	seg.setInputNormals(cloud_normals2);

	// obtain the cylinder inliers and coefficients
	seg.segment(*inliers_cylinder ,*coefficients_cylinder);
	std::cerr << "Cylinder coefficients : " << *coefficients_cylinder << std::endl;
	extract.setInputCloud(cloud_filtered2);
	extract.setIndices(inliers_cylinder);
	extract.setNegative(false);
	pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>);
	extract.filter(*cloud_cylinder);
	if (cloud_cylinder->points.empty()) {
		std:cerr << "Can not find the cylindrical component." << std::endl;
	}
	else {
		std::cerr << "Point cloud find the cylindrical component." << std::endl;
		writer.write("table_scene_mug_stereo_textured_cylinder.pcd" ,*cloud_cylinder);
	}

	// view
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
	int v1(1);
	int v2(2);
	int v3(3);
	viewer->createViewPort(0.0 ,0.0 ,0.5 ,1.0 ,v1);
	viewer->createViewPort(0.5 ,0.0 ,1.0 ,0.5 ,v2);
	viewer->createViewPort(0.5 ,0.5 ,1.0 ,1.0 ,v3);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_out_red(cloud ,255 ,0 ,0);
	viewer->addPointCloud(cloud ,cloud_out_red ,"cloud_out1" ,v1);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_out_green(cloud_plane ,0 ,255 ,0);
	viewer->addPointCloud(cloud_plane, cloud_out_green ,"cloud_out2" ,v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_out_blue(cloud_cylinder ,0 , 0 ,255);
	viewer->addPointCloud(cloud_cylinder ,cloud_out_blue ,"cloud_out3" ,v3);

	while (!viewer->wasStopped()) {
		viewer->spinOnce();
		boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}

	return 0;
}
