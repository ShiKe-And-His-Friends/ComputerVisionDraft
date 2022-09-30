/***
	Visualization 可视化 基础
**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

typedef pcl::PointXYZ PointType;

int visualization_cat_horse(int argc ,char **argv) {

	std::cout << " visualization base." << std::endl;

	pcl::PointCloud<PointType>::Ptr cloud1(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr cloud2(new pcl::PointCloud<PointType>);
	pcl::io::loadPCDFile("ism_train_cat.pcd" ,*cloud1);
	pcl::io::loadPCDFile("ism_train_horse.pcd", *cloud2);

	pcl::visualization::PCLVisualizer viewer;
	viewer.setBackgroundColor(0 ,0 ,0);
	viewer.addPointCloud(cloud1,"cloud1");
	pcl::visualization::PointCloudColorHandlerCustom<PointType> red(cloud2 ,255 ,0 ,0 );
	viewer.addPointCloud(cloud2, "cloud2");

	PointType temp1 = cloud1->points[0];
	PointType temp2 = cloud1->points[1];
	viewer.addLine(temp1, temp2, 0, 255, 0, "line0");

	viewer.spin();

	while (!viewer.wasStopped()) {
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}