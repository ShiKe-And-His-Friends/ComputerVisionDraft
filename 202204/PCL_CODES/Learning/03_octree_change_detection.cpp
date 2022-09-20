#include <iostream>
#include <pcl/point_types.h>
#include <pcl/octree/octree.h>

int octree_change_detection(int argc ,char ** argv) {
	std::cout << "octree change detection." << std::endl;

	float resolution = 32.0f;
	pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree(resolution);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
	cloudA->width = 128;
	cloudA->height = 1;
	cloudA->points.resize(cloudA->width * cloudA->height);

	for (size_t i = 0; i < cloudA->size(); i++) {
		cloudA->points[i].x = 64.0f * rand() / (RAND_MAX + 1.0f);
		cloudA->points[i].y = 64.0f * rand() / (RAND_MAX + 1.0f);
		cloudA->points[i].z = 64.0f * rand() / (RAND_MAX + 1.0f);
	}
	octree.setInputCloud(cloudA);
	octree.addPointsFromInputCloud();

	// memeory buffer usage
	octree.switchBuffers();

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
	cloudB->width = 128;
	cloudB->height = 1;
	cloudB->points.resize(cloudB->width * cloudB->height);

	for (size_t i = 0; i < cloudB->size(); i++) {
		cloudB->points[i].x = 64.0f * rand() / (RAND_MAX + 1.0f);
		cloudB->points[i].y = 64.0f * rand() / (RAND_MAX + 1.0f);
		cloudB->points[i].z = 64.0f * rand() / (RAND_MAX + 1.0f);
	}
	octree.setInputCloud(cloudB);
	octree.addPointsFromInputCloud();

	// check new add point frome voxel
	std::vector<int> newPointIdxVector;
	octree.getPointIndicesFromNewVoxels(newPointIdxVector); // A && !B

	for (size_t i = 0; i < newPointIdxVector.size(); i++) {
		std::cout << cloudB->points[newPointIdxVector[i]].x << " "
			<< cloudB->points[newPointIdxVector[i]].y << " "
			<< cloudB->points[newPointIdxVector[i]].z << " "
			<< std::endl;
	}

	return 0;
}