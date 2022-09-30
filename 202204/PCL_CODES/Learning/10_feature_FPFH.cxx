/***
	FPFH 快速点特征直方图
	Faster Point Feature Histogram
*/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_plotter.h>

int feature_faster_point_histogram(int argc ,char **argv) {

	std::cout << "feature faster point feature hoistogram." << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("ism_test_cat.pcd", *cloud);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	normals->points.resize(cloud->points.size());

	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(cloud);
	fpfh.setInputNormals(normals);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	fpfh.setSearchMethod(tree);

	// output FPFH sedcriptors
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);
	fpfh.setRadiusSearch(0.05);
	fpfh.compute(*fpfhs);

	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*fpfhs, 300);
	plotter.plot();

	return 0;
}