#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>


int kdtree_search(int argc ,char ** argv) {

	std::cout << "kdtree example" << std::endl;
	srand(time(NULL));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	cloud->width = 1000;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}

	// use FLANN alogrithem
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	int K = 10; // set threshold
	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);

	// nearest search
	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
		for (size_t i = 0; i < K; i++) {
			std::cout << cloud->points[pointIdxNKNSearch[i]].x << " ";
			std::cout << cloud->points[pointIdxNKNSearch[i]].y << " ";
			std::cout << cloud->points[pointIdxNKNSearch[i]].z << " " ;
			std::cout << pointNKNSquaredDistance[i] << std::endl;
		}
		std::cout << "////////// Nearest Search //////////////" << std::endl;
	}
	else {
		std::cout << "FLANN search nearset failure." << std::endl;
	}

	// radius search
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointIdxRadiusSquaredDistance;
	float radius = 256.0f * rand() / (RAND_MAX + 1.0f);
	if (kdtree.radiusSearch(searchPoint ,radius ,pointIdxRadiusSearch ,pointIdxRadiusSquaredDistance) > 0) {
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); i++) {
			std::cout << cloud->points[pointIdxRadiusSearch[i]].x << " ";
			std::cout << cloud->points[pointIdxRadiusSearch[i]].y << " ";
			std::cout << cloud->points[pointIdxRadiusSearch[i]].z << " ";
			std::cout << pointIdxRadiusSquaredDistance[i] << std::endl;
		}
		std::cout << "////////// Radius Search //////////////" << std::endl;
	}
	else {
		std::cout << "FLANN search radius failure." << std::endl;
	}

	return 0;
}