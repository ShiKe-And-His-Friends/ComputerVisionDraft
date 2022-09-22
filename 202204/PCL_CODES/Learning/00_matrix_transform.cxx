#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Eigen>

void showHelp(int argc ,char **argv) {
	std::cout << std::endl;
	std::cout << "Usage: " << argv[0] << " cloud_filename.[pcd|ply]" << std::endl;
	std::cout << "-h: show this help" << std::endl;
}

int main00(int argc ,char **argv) {

	if (pcl::console::find_switch(argc, argv, "-h") || pcl::console::find_switch(argc,argv ,"-help")) {
		showHelp(argc ,argv);
		return 0;
	}

	std::vector<int> filenames;
	bool file_is_pcd = false;
	
	// input file ref
	filenames = pcl::console::parse_file_extension_argument(argc,argv ,".ply");
	if (filenames.size() != 1) {
		filenames = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
		if (filenames.size() != 1) {
			showHelp(argc ,argv);
			return -1;
		}
		else {
			file_is_pcd = true;
		}
	}

	// load file
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	if (file_is_pcd) {
		if (pcl::io::loadPCDFile(argv[filenames[0]] ,*source_cloud)) {
			std::cout << "decode pcd file error." << std::endl;
			showHelp(argc, argv);
			return -1;
		}
	}
	else {
		if (pcl::io::loadPLYFile(argv[filenames[0]], *source_cloud)) {
			std::cout << "decode ply file error." << std::endl;
			showHelp(argc, argv);
			return -1;
		}
	}

	// matrix
	// |  1  0  0  x  |
	// |  0  1  0  y  |
	// |  0  0  1  z  |
	// |  0  0  0  1  |   transform vector {x ,y ,z}
	Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
	
	float theta = M_PI / 4;
	transform_1(0, 0) = std::cos(theta);
	transform_1(0, 1) = -std::sin(theta);
	transform_1(1, 0) = std::sin(theta);
	transform_1(1, 1) = std::cos(theta);
	transform_1(0, 3) = 2.5f;
	std::cout << "matrix transform" << std::endl;
	std::cout << transform_1 << std::endl;
	

	// Affine
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	transform_2.translation() << 2.5, 0.0, 0.0;
	transform_2.rotate(Eigen::AngleAxisf(theta ,Eigen::Vector3f::UnitZ()));
	std::cout << "affine transform" << std::endl;
	std::cout << transform_2.matrix() << std::endl;

	// transform
	pcl::PointCloud<pcl::PointXYZ>::Ptr transform_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*source_cloud ,*transform_cloud ,transform_2);

	// visualization
	pcl::visualization::PCLVisualizer viewer("Matrix transform");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler(source_cloud ,255 ,0 ,0);
	viewer.addPointCloud(source_cloud ,source_cloud_color_handler ,"original_cloud");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler(transform_cloud, 0, 255, 0);
	viewer.addPointCloud(transform_cloud, transformed_cloud_color_handler, "transformed_cloud");
	viewer.addCoordinateSystem(1.0 ,"cloud" ,0);
	viewer.setBackgroundColor(0.05 ,0.05 ,0.05 ,0);
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,2 ,"original_cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "transformed_cloud");

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;
}