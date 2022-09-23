/***
	使用参数化模型投影
**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>

int project_inlies(int agrc ,char **agrv) {

	std::cout << "filter project inliers." << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_project(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 5;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
	}
	std::cerr << "Cloud before projections: " << std::endl;
	for (size_t i = 0; i < cloud->points.size();i++) {
	std::cerr << "  " << cloud->points[i].x << " "
		<< cloud->points[i].y << " "
		<< cloud->points[i].z << " " << std::endl;
	}

	// 填充ModelConefficients的值 ax + by + cz + d = 0
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	pcl::ProjectInliers<pcl::PointXYZ> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(cloud);
	proj.setModelCoefficients(coefficients);
	proj.filter(*cloud_project);

	std::cerr << "Cloud after projections: " << std::endl;
	for (size_t i = 0; i < cloud_project->points.size(); i++) {
		std::cerr << "  " << cloud_project->points[i].x << " "
			<< cloud_project->points[i].y << " "
			<< cloud_project->points[i].z << " " << std::endl;
	}

	return 0;
}