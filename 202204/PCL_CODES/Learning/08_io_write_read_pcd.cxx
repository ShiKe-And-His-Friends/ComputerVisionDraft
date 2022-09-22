#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int io_write_read_pcd(int agrc ,char **argv) {

	// PCL write
	std::cout << "write and read pcd " << std::endl;
	pcl::PointCloud<pcl::PointXYZ> cloud;
	cloud.width = 5;
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width * cloud.height);
	for (size_t i = 0; i < cloud.points.size();i++) {
		cloud.points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud.points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud.points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
	}
	pcl::io::savePCDFile("test_io_pcd.pcd" ,cloud);
	std::cerr << "Saved " << cloud.points.size() << "data points to test_io_pcd.pcd" << std::endl;
	for (size_t i = 0; i < cloud.points.size() ;i++) {
		std::cerr << " " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;
	}
	cloud.resize(0);

	// PCL read
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile("test_io_pcd.pcd", *cloud_in) == -1) {
		std::cerr << "Read data points to test_io_pcd.pcd failure." << std::endl;
		return -1;
	}
	std::cerr << "Read " << cloud_in->points.size() << "data points to test_io_pcd.pcd" << std::endl;
	for (size_t i = 0; i < cloud_in->points.size(); i++) {
		std::cerr << " " << cloud_in->points[i].x << " " << cloud_in->points[i].y << " " << cloud_in->points[i].z << std::endl;
	}

	return 0;
}