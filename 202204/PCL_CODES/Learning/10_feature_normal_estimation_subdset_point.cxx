/***
	估计点云子集的表面法线
**/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>

int feature_normal_estimation_subdest_point(int argc ,char **argv) {

	std::cout << "" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("table_scene_lms400.pcd", *cloud);

	std::vector<int> indices(std::floor(cloud->points.size() )); // 用前10%的点
	for (std::size_t i = 0; i < indices.size(); i++) {
		indices[i] = i;
	}
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);

	// 传递索引
	pcl::shared_ptr<std::vector<int>> indicesptr(new std::vector<int>(indices));
	ne.setIndices(indicesptr);

	// 创建一个空的kdtree，用来normal estimatio搜索
	// 它的内容根据输入数据填充到对象内部（因为没提供其他搜索平面 ）
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud < pcl::Normal> );
	ne.setRadiusSearch(0.03);
	ne.compute(*cloud_normals);

	// 储存点云特征
	pcl::PCDWriter writer;
	writer.write<pcl::Normal>("table_cloud_normals_subdest.pcd", *cloud_normals, false);

	// 可视
	pcl::visualization::PCLVisualizer viewer("PCL viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals);

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;
}