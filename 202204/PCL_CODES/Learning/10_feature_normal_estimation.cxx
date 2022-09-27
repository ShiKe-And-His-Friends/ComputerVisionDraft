/***
	评估整个点云表面的法向量
**/
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>

int feature_normal_estimation(int argc ,char *argv) {

	std::cout << "feature normal estimation." << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("table_scene_lms400.pcd" ,*cloud);
	std::cout << "points " << cloud->points.size() << std::endl;

	// 法线估计向量
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	pcl::search::KdTree < pcl::PointXYZ >::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.03); //3厘米半径的元素
	ne.compute(*cloud_normals);

	// 储存点云特征
	pcl::PCDWriter writer;
	writer.write<pcl::Normal>("table_cloud_normals.pcd",*cloud_normals ,false);

	// 可视
	pcl::visualization::PCLVisualizer viewer("PCL viewer");
	viewer.setBackgroundColor( 0 , 0 ,0);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud , cloud_normals);

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;
}