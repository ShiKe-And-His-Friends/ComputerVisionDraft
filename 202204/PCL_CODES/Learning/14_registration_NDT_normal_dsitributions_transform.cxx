/****

	��̬�ֲ��任ƥ�� NDT

***/
#include <iostream>
#include <Eigen>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

int registration_normal_distributed(int argc ,char **argv) {

	std::cout << "registration normal distributed tranform." << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("room_scan1.pcd",*target_cloud) == -1 ) {
		PCL_ERROR("cloud not read file room_scan1.pcd\n");
		return -1;
	}
	std::cout << "Loaded " << target_cloud->size() << " data points from room1 pcd" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("room_scan2.pcd", *input_cloud) == -1) {
		PCL_ERROR("cloud not read file room_scan2.pcd\n");
		return -1;
	}
	std::cout << "Loaded " << input_cloud->size() << " data points from room2 pcd" << std::endl;

	// �������ɨ��������ݹ��˵�ԭʼ�ߴ��10%�����ƥ���ٶ�
	// NDT�㷨��Ŀ����ƵĶ�Ӧ��������ṹͳ�Ʋ����õ������أ���ʹ�����ء��ϲ����˲�Ӱ�쾫��ͳ��
	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
	approximate_voxel_filter.setLeafSize(0.2 ,0.2 ,0.2);
	approximate_voxel_filter.setInputCloud(input_cloud);
	approximate_voxel_filter.filter(*filtered_cloud);
	std::cout << "Filtered cloud contains " << filtered_cloud->size() << std::endl;

	//��ʼ��NDT����
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	ndt.setTransformationEpsilon(0.01);//��ֹ��������С����
	ndt.setStepSize(0.1); //more-thuente��������󲽳�
	ndt.setResolution(1.0); //NDT����ֱ���voxelgridecovariance
	ndt.setMaximumIterations(35);
	ndt.setInputSource(filtered_cloud);
	ndt.setInputTarget(target_cloud);

	// ���ʹ�û����˲�෨�õ��Ĵ��Գ�ʼ�任������
	Eigen::AngleAxisf init_rotation(0.6931, Eigen::Vector3f::UnitZ());
	Eigen::Translation3f init_translation(1.79387 ,0.720047 ,0);
	Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();
	std::cout << init_guess << std::endl;

	// ������Ҫ����任���Ա�����Դ����ƥ�䵽Ŀ�����
	pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	ndt.align(*output_cloud ,init_guess);
	std::cout << "Normal Distributions Transform has converaged: " << ndt.hasConverged()
		<< " score:" << ndt.getFitnessScore() << std::endl;
	pcl::transformPointCloud(*input_cloud ,*output_cloud ,ndt.getFinalTransformation());
	pcl::io::savePCDFileASCII("room_scan2_transformed_20221008.pcd" ,*output_cloud);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_final(new pcl::visualization::PCLVisualizer("3D viewer"));
	viewer_final->setBackgroundColor(0 ,0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud ,255 ,0 ,0);
	viewer_final->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,1 ,"target cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(output_cloud ,0 ,255 ,0);
	viewer_final->addPointCloud<pcl::PointXYZ>(output_cloud ,output_color ,"output cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,1,"output cloud");

	viewer_final->addCoordinateSystem(1.0);
	viewer_final->initCameraParameters();
	while (!viewer_final->wasStopped()) {
		viewer_final->spinOnce(1000);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}