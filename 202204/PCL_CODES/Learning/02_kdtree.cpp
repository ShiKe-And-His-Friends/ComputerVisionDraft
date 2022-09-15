#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>


int main(int argc ,char ** argv) {

	std::cout << "kdtree example" << std::endl;
	srand(time(NULL));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	cloud->width = 1000;
	cloud->height = 1;

	return 0;
}