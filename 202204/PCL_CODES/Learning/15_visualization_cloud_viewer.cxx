/***
	Visualization 

	Cloud Viewer + Text
**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>

typedef pcl::PointXYZRGBA PointType;

static int user_data;

void viewerOnceOff(pcl::visualization::PCLVisualizer &viewer) {
	viewer.setBackgroundColor(1.0 ,0.5 ,1.0);
	pcl::PointXYZ o;
	o.x = 1.0f;
	o.y = .0f;
	o.z = .0f;
	viewer.addSphere(o ,0.25 ,"sphere" ,0);
	std::cout << "i only run once" << std::endl;
}

void viewerPsycho(pcl::visualization::PCLVisualizer &viewer) {
	static unsigned count = 0;
	std::stringstream ss;
	ss << "Once per value loop:" << count++ ;
	viewer.removeShape("text" ,0);
	viewer.addText(ss.str() ,200 ,300 ,"text" ,0);

	user_data++;
}

int visualization_cloud_viewer(int argc, char ** argv) {

	std::cout << "visualization cloud viewer." << std::endl;

	pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
	pcl::io::loadPCDFile("room_scan1.pcd" ,*cloud);

	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloud);

	// regist once
	viewer.runOnVisualizationThreadOnce(viewerOnceOff);

	// draw everytimes
	viewer.runOnVisualizationThread(viewerPsycho);

	while (!viewer.wasStopped()) {
		user_data++;
		boost::this_thread::sleep(boost::posix_time::milliseconds(100000));
	}

	return 0;
}