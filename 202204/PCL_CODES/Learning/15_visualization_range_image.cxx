/***
	Visualization

	Range Image

	## important show range informations

**/
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <boost/thread/thread.hpp>

namespace Visualization_Range_Image {

	typedef pcl::PointXYZ PointType;

	float angular_resolution_x = 0.5;
	float angular_resolution_y = 0.5;
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
	bool live_update = false;

	void printHelp(char* console_dir) {

		std::cout <<
			"\nUsage : " << console_dir << " <option> [sample.pcd]"
			<< "-rx <float> angular resolution in degress ,default" << angular_resolution_x << " \n"
			<< "-ry <float> angular resolution in degress ,default" << angular_resolution_y << " \n"
			<< "-c <int> coordinate frame ,default" << (int)coordinate_frame << " \n"
			<< "-l live update the range image according to the selected view in the 3D viewer \n"
			<< "-h help\n"
			<< "\n"
			<< std::endl;
	}

	void setViewerPose(pcl::visualization::PCLVisualizer &viewer ,const Eigen::Affine3f &viewer_pose) {
		Eigen::Vector3f pose_vector = viewer_pose * Eigen::Vector3f(0 ,0 ,0);
		Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pose_vector;
		Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0 ,-1 ,0);
		viewer.setCameraPosition(
			pose_vector[0], pose_vector[1], pose_vector[2],
			look_at_vector[0], look_at_vector[1], look_at_vector[2],
			up_vector[0], up_vector[1], up_vector[2]
		);
	}

};

using namespace Visualization_Range_Image;

int visualiztion_range_image(int argc ,char **argv) {

	std::cout << "visualization cloud viewer" << std::endl;

	if (pcl::console::find_argument(argc ,argv ,"-h") >= 0) {
		printHelp(argv[0]);
		return 0;
	}
	if (pcl::console::find_argument(argc ,argv ,"-l") >= 0) {
		live_update = true;
		std::cout << "Live update is on." << std::endl;
	}
	if (pcl::console::parse(argc ,argv ,"-rx" ,angular_resolution_x) >= 0) {
		std::cout << "Setting angular resolution in x-direction to " << angular_resolution_x << std::endl;
	}
	if (pcl::console::parse(argc, argv, "-ry", angular_resolution_y) >= 0) {
		std::cout << "Setting angular resolution in y-direction to " << angular_resolution_y << std::endl;
	}
	int temp_coordinate_frame;
	if (pcl::console::parse(argc ,argv ,"-c" ,temp_coordinate_frame) >= 0) {
		coordinate_frame = pcl::RangeImage::CoordinateFrame(temp_coordinate_frame);
		std::cout << "Using coordinate frame " << (int)coordinate_frame << std::endl;
	}
	angular_resolution_x = pcl::deg2rad(angular_resolution_x);
	angular_resolution_y = pcl::deg2rad(angular_resolution_y);
	
	// create point cloud
	pcl::PointCloud<PointType>::Ptr point_cloud_ptr(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
	Eigen::Affine3f scen_sensor_pose(Eigen::Affine3f::Identity());
	std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument(argc ,argv ,"pcd");

	if (!pcd_filename_indices.empty()) {
		std::string filename = argv[pcd_filename_indices[0]];
		if (pcl::io::loadPCDFile(filename ,point_cloud) == -1) {
			std::cout << "read pcd file error: " << filename << std::endl;
			printHelp(argv[0]);
			return -1;
		}
		scen_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(point_cloud.sensor_origin_[0],
			point_cloud.sensor_origin_[1], 
			point_cloud.sensor_origin_[2])) *
			Eigen::Affine3f(point_cloud.sensor_orientation_);
	}
	else {
		std::cout << "No pcd file ,given random point cloud." << std::endl;
		for (float x = -.05f; x < 0.5f ;x+=0.01f) {
			for (float y = -.05f; y < 0.5f; y += 0.01f) {
				PointType point;
				point.x = x;
				point.y = y;
				point.z = 2.0f - y;
				point_cloud.points.push_back(point);
			}
		}
		point_cloud.width = (int)point_cloud.points.size();
		point_cloud.height = 1;
	}

	// createe range image from point cloud
	float noise_level = 0.0;
	float min_range = 0.0f;
	int border_size = 1;
	boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
	pcl::RangeImage& range_image = *range_image_ptr;
	range_image.createFromPointCloud(
		point_cloud ,angular_resolution_x ,angular_resolution_y,
		pcl::deg2rad(360.0f) ,pcl::deg2rad(180.0f),
		scen_sensor_pose ,coordinate_frame ,noise_level ,min_range ,border_size
	);

	// open 3d viewer
	pcl::visualization::PCLVisualizer viewer("3D viewer");
	viewer.setBackgroundColor(0 ,0,0);
	// pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> r_color_handler(range_image_ptr ,0 ,0 ,0);
	// viewer.addPointCloud(range_image_ptr, range_image_color_handler, "range image");
	pcl::visualization::PointCloudColorHandlerCustom<PointType> point_cloud_color_handler(point_cloud_ptr ,255 ,0 ,0);
	viewer.addPointCloud(point_cloud_ptr, point_cloud_color_handler,"range image");
	viewer.addCoordinateSystem(1.0f ,"global");
	viewer.initCameraParameters();
	setViewerPose(viewer ,range_image.getTransformationToWorldSystem());

	// show range image
	pcl::visualization::RangeImageVisualizer range_image_widget("Range image");
	range_image_widget.showRangeImage(range_image);

	while (!viewer.wasStopped()) {
		range_image_widget.spinOnce(50);
		viewer.spinOnce(50);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));

		if (live_update) {
			scen_sensor_pose = viewer.getViewerPose();
			range_image.createFromPointCloud(
				point_cloud, angular_resolution_x ,angular_resolution_y,
				pcl::deg2rad(360.0f ),pcl::deg2rad(180.0f),
				scen_sensor_pose ,pcl::RangeImage::LASER_FRAME ,noise_level ,min_range ,border_size
			);
			range_image_widget.showRangeImage(range_image);
		}
	}

	return 0;
}

