#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

int point_cloud_passthrough(int argc ,char **argv) {
	std::cout << "filters pass through" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 5;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size();i++) {
		cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
	}
	std::cerr << "Cloud before filtering: " << std::endl;
	for (size_t i = 0; i < cloud->points.size(); i++) {
		std::cerr << "  " << cloud->points[i].x << " "
			<< cloud->points[i].y << " "
			<< cloud->points[i].z << " " << std::endl;
	}

	// 创建直通滤波，方向Z轴，范围(0 ,1)
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0 ,800.0);
	//pass.setFilterLimitsNegative(true);
	pass.filter(*cloud_filtered);

	std::cerr << "Cloud after filter: " << std::endl;
	for (size_t i = 0; i < cloud_filtered->points.size();i++) {
		std::cerr << "  " << cloud_filtered->points[i].x << " "
			<< cloud_filtered->points[i].y << " "
			<< cloud_filtered->points[i].z << " " << std::endl;
	}

	return 0;
}