/***
	��һ����������ȡ����
**/
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

int filter_extract_indices(int argc,char **argv) {

	std::cout << "extract indices." << std::endl;

	// voxel-box �˲����²��������ٴ����ٶ�
	pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2) ,cloud_filtered_blob(new pcl::PCLPointCloud2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>)
		, cloud_p(new pcl::PointCloud<pcl::PointXYZ>)
		, cloud_f(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PCDReader reader;
	reader.read("table_scene_lms400.pcd" ,*cloud_blob);
	std::cerr << "Point cloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points," << std::endl;

	// voxel-box filter
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud_blob);
	sor.setLeafSize(0.01f ,0.01f ,0.01f);
	sor.filter(*cloud_filtered_blob);

	// ת��ģ�����
	pcl::fromPCLPointCloud2(*cloud_filtered_blob ,*cloud_filtered);
	std::cerr << "Point cloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points," << std::endl;
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>("table_scene_lms400_downsampled.pcd" ,*cloud_filtered ,false);

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::SACSegmentation<pcl::PointXYZ>  seg; //�����ָ����
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(1000); //����������
	seg.setDistanceThreshold(0.01);

	// ����ExtractIndicesʵ�ʲ��� 
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	int i = 0, nr_points = (int)cloud_filtered->points.size();
	// 30%��ԭʼ�����
	while (cloud_filtered->points.size() > 0.3 * nr_points) {
		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0) {
			std::cerr << "Cloud not estimate a planar model for the given dataset." << std::endl;
			break;
		}
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*cloud_p);
		std::cerr << "PointCloud representing the planar component: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

		std::stringstream ss;
		ss << "table_scene_lms400_plane_" << i << ".pcd";
		writer.write<pcl::PointXYZ>(ss.str() ,*cloud_p ,false);

		// �����˲�����
		extract.setNegative(true);
		extract.filter(*cloud_f);
		cloud_filtered.swap(cloud_f);
		i++;
	}

	return 0;
}