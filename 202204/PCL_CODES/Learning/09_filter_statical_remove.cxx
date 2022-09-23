/***
	离群点滤除
**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

int filter_statical_remove(int argc ,char ** argv) {

	std::cout << "statistacal remove" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PCDReader reader;
	reader.read<pcl::PointXYZ>("table_scene_lms400.pcd" ,*cloud);
	std::cerr << "cloud before filtering: " << std::endl;
	std::cerr << *cloud << std::endl;

	// 创建滤波器，邻近点个数50，标准差1（超过1个距离是离群点）
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*cloud_filtered);

	std::cerr << "Cloud after filtering: " << std::endl;
	std::cerr << *cloud_filtered << std::endl;

	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>("table_scene_lms400_inliers.pcd" ,*cloud_filtered ,false);

	sor.setNegative(true);
	writer.write<pcl::PointXYZ>("table_scene_lms400_outliers.pcd", *cloud_filtered ,false);

	return 0;
}