/***

	Visualizaiton    
	small sample

**/

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

int visualization_small_sample(int argc ,char **argv) {

	std::cout << "visualization small sample." << std::endl;

	pcl::visualization::PCLVisualizer viewer;
	viewer.setBackgroundColor(0,0,0);
	// red green blue coordinate
	viewer.addCoordinateSystem(3.0);
	viewer.initCameraParameters();

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("ism_train_lioness.pcd" ,*cloud);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(cloud ,255 ,0 ,0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud ,rgb ,"sample cloud33");
	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;
}