#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

int main(int argc, char** argv) {
	std::cout << "voxel box grid" << std::endl;

	pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
	pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());
	pcl::PCDReader reader;
	reader.read("table_scene_lms400.pcd" ,*cloud);
	std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
		<< " data points (" << pcl::getFieldsList(*cloud) << ").";
	// 创建一个voxel叶大小1cm的VoxelGrid滤波器
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.1f, 0.1f, 0.1f);
	sor.filter(*cloud_filtered);
	std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
		<< " data points (" << pcl::getFieldsList(*cloud_filtered) << ").";
	pcl::PCDWriter writer;
	writer.write("table_scene_lms400_downsample_chaneg.pcd" ,*cloud_filtered ,
		Eigen::Vector4f::Zero() ,Eigen::Quaternionf::Identity() ,false);

	return 0;
}