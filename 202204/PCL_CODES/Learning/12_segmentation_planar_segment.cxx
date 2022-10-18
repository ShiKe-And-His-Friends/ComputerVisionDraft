/***

	分割  基于平面

**/
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread.hpp>

int segmentation_planar(int argc, char** argv) {

	std::cout << "Segmentation planar segmentation." << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 15;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);

	for (size_t i = 0; i < cloud->points.size();i++) {
		cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1.0;
	}

	// random set outliners points
	cloud->points[0].z = 2.0;
	cloud->points[3].z = -2.0;
	cloud->points[6].z = 4.0;
	std::cerr << "Point cloud data: " << cloud->points.size() << " " << std::endl;

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.01);
	seg.setInputCloud(cloud);
	seg.segment(*inliers ,*coefficients);
	if (inliers->indices.size() == 0) {
		PCL_ERROR("Cloud not estimate a planar model for the given dataset.");
		return -1;
	}
	std::cerr << "Model coefficiants: " << coefficients->values[0] << " "
		<< coefficients->values[1] << " "
		<< coefficients->values[2] << " "
		<< coefficients->values[3] << " "
		<< std::endl;
	
	pcl::visualization::PCLVisualizer viewer("planar segmentation viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud ,255 ,0 ,0);
	viewer.addPointCloud(cloud ,cloud_color_handler ,"point clouds");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,7 ,"point clouds");

	if (!viewer.wasStopped()) {
		viewer.spin();
		boost::this_thread::sleep(boost::posix_time::microseconds(1000));
	}

	return 0;
}
