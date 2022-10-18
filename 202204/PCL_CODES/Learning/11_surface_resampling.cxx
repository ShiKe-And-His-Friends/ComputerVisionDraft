/***

	surface表面  重采样

**/

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/surface/mls.h>
#include <pcl/visualization/cloud_viewer.h>

int surface_downsampling() {
	std::cout << "Surface resampling." << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile("ism_train_cat.pcd" ,*cloud) == -1) {
		std::cerr << "load point cloud file error." << std::endl;
		return -1;
	}
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointNormal> mls_points;

	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
	mls.setComputeNormals(true);
	mls.setInputCloud(cloud);
	mls.setPolynomialOrder(2);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(0.03);
	mls.process(mls_points); // reconstruct

	if (mls_points.size() > 0) {
		pcl::io::savePCDFile("target-mls.pcd" ,mls_points);
	}
	else {
		std::cout << "保存数据为空" << std::endl;
	}

	pcl::visualization::CloudViewer viewer("cloud viewer");
	viewer.showCloud(cloud);
	while (!viewer.wasStopped()) {
	
	}

	return 0;
}