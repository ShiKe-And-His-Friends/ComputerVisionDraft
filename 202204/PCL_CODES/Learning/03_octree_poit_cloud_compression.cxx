#include <iostream>
#include <ctime>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>

int spatial_partioning_search_operation(int argc ,char **argv) {
	
	std::cout << "octree spatial partioning and search operation." << std::endl;

	srand((unsigned int)time(NULL));

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 1000;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}

	// create octree
	float resolution = 128.0f;
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud(); // make

	pcl:: PointXYZ searchPoint;
	std::vector<int> pointIdxVec;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	if (octree.voxelSearch(searchPoint ,pointIdxVec)) {
		for (size_t i = 0; i < pointIdxVec.size(); i++) {
			std::cout << cloud->points[pointIdxVec[i]].x << " "
				<< cloud->points[pointIdxVec[i]].y << " "
				<< cloud->points[pointIdxVec[i]].z << " "
				<< std::endl;
		}
	}

	return 0;
}