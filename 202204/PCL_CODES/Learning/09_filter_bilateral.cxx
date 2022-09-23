/***	
	Ë«±ßÂË²¨
**/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/impl/bilateral.hpp>
#include <pcl/search/kdtree.h>

void bilateralFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr &input , pcl::PointCloud<pcl::PointXYZI>::Ptr& output) {
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZI>);
	pcl::BilateralFilter<pcl::PointXYZI> fbf;
	fbf.setInputCloud(input);
	fbf.setSearchMethod(tree1);
	fbf.setStdDev(0.1);
	fbf.setHalfSize(0.1);
	fbf.filter(*output);
}

int filter_bilateral_(int agrc ,char **argv) {

	std::cout << "filter bilateral " << std::endl;

	pcl::PointCloud < pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud < pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);

	pcl::PCDReader reader;
	reader.read<pcl::PointXYZI>("table_scene_lms400.pcd", *cloud);
	bilateralFilter(cloud ,cloud_filtered);
	
	pcl::PCDWriter writer;
	writer.write("table_scene_lms400_bilateral_chaneg.pcd", *cloud_filtered, false);

	return 0;
}