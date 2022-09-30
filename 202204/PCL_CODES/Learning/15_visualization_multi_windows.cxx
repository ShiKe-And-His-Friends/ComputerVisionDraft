/***
	Visualizaiton
	multi windows
**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <boost/thread/thread.hpp>

int visualization_multi_windows(int argc, char** argv) {

	std::cout << "visualizaiton two windows" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ>("table_scene_mug_stereo_textured.pcd", *cloud);
	std::cout << "cloud before filtered" << std::endl;
	std::cout << *cloud << std::endl;

	pcl::visualization::PCLVisualizer viewer("Two windows");
	int v1(0);
	int v2(1);

	viewer.createViewPort(0.0, 0.0, 0.5, 1, v1);
	viewer.setBackgroundColor(0, 0, 0, v1);
	viewer.createViewPort(0.5, 0.0, 1, 1, v2);
	viewer.setBackgroundColor(0.5, 0.5, 0.5, v2);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_out_blue(cloud, 0, 0, 255);
	viewer.addPointCloud(cloud, cloud_out_blue, "cloud_out1", v1);

	pcl::PassThrough<pcl::PointXYZ> pass; //create filter
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0, 1.5);
	// pass.setFilterLimitsNegative(true);
	pass.filter(*cloud_filtered);
	std::cout << "cloud after filtered" << std::endl;
	std::cout << *cloud_filtered << std::endl;
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_out_orage(cloud_filtered, 250, 128, 10);
	viewer.addPointCloud(cloud_filtered, cloud_out_orage, "cloud_out2", v2);

	while (!viewer.wasStopped()) {
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		viewer.spinOnce(200);
	}

	return 0;
}