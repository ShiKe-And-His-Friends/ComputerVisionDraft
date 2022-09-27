/***
	���Ƶ����Ӽ��ı��淨��
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

	std::vector<int> indices(std::floor(cloud->points.size() )); // ��ǰ10%�ĵ�
	for (std::size_t i = 0; i < indices.size(); i++) {
		indices[i] = i;
	}
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);

	// ��������
	pcl::shared_ptr<std::vector<int>> indicesptr(new std::vector<int>(indices));
	ne.setIndices(indicesptr);

	// ����һ���յ�kdtree������normal estimatio����
	// �������ݸ�������������䵽�����ڲ�����Ϊû�ṩ��������ƽ�� ��
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud < pcl::Normal> );
	ne.setRadiusSearch(0.03);
	ne.compute(*cloud_normals);

	// �����������
	pcl::PCDWriter writer;
	writer.write<pcl::Normal>("table_cloud_normals_subdest.pcd", *cloud_normals, false);

	// ����
	pcl::visualization::PCLVisualizer viewer("PCL viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals);

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;
}