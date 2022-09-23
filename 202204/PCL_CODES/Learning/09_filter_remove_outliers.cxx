/***
	删除不符合用户指定的一个或多个数据点
	半径
	高度宽度的逻辑组合
**/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

int fliter_remove_outliers(int argc ,char ** argv) {

	bool remove_signle = false;

	std::cout << "filter remove outlisers" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 5;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
		cloud->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
	}
	if (remove_signle) {
		pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
		outrem.setInputCloud(cloud);
		outrem.setRadiusSearch(0.7 * 1024);
		outrem.setMinNeighborsInRadius(2);
		outrem.filter(*cloud_filtered);
	}
	else {
		// 条件限定滤波器
		pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>);
		// GT EQ LT GE LE
		range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison < pcl::PointXYZ>("z" ,pcl::ComparisonOps::GT ,0.0)));
		range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison < pcl::PointXYZ>("z", pcl::ComparisonOps::LT, 0.8)));
		pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
		condrem.setCondition(range_cond);
		condrem.setInputCloud(cloud);
		condrem.setKeepOrganized(true); //保证点云结构的有序性
		condrem.filter(*cloud_filtered);
	}

	std::cerr << "Cloud before filtering: "  << std::endl;
	for (size_t i = 0; i < cloud->points.size(); i++) {
		std::cerr << "  " << cloud->points[i].x << " "
			<< cloud->points[i].y << " "
			<< cloud->points[i].z << " " << std::endl;
	}
	std::cerr << "Cloud after filtering: " << std::endl;
	for (size_t i = 0; i < cloud_filtered->points.size(); i++) {
		std::cerr << "  " << cloud_filtered->points[i].x << " "
			<< cloud_filtered->points[i].y << " "
			<< cloud_filtered->points[i].z << " " << std::endl;
	}
	return 0;
}