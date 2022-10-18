/*********** ERROR *************
	PCL 1.12.1 makes crash
	range_image_border_widght = pcl::visualization::RangeImageVisualizer::getRangeImageBorderWidget(range_image ,...);

	#include <vtkAutoInit.h>
	VTK_MODULE_INIT(vtkRenderingContextOpenGL2)
 **/
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/file_io.h>
#include <pcl/features/range_image_border_extractor.h>
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2)

namespace Range_image_create {

	typedef pcl::PointXYZ PointType;
	// parameters
	float angular_resolution = 0.5f;
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
	bool setUnseenToMaxRange = false;

	void printHelp(const char* progName) {
		std::cout << "\nUsage: " << progName << " [options] <scene.pcd>" << "\n"
			<< "\t  -r <float> angular resolution in degress (default " << angular_resolution << ") \n"
			<< "\t  -c <int> coordinate frame (default " << (int)coordinate_frame << ") \n"
			<< "\t  -m Treat all unseen points to max range. " << "\n"
			<< "\t  -h Help" << "\n"
			<< "\t  " << "\n"
			<< std::endl;
	}

};

using namespace Range_image_create;

int main222(int argc, char** argv) {

	std::cout << "Range image create." << std::endl;
	if (pcl::console::find_argument(argc, argv, "-h") >= 0) {
		printHelp(argv[0]);
		return -1;
	}
	if (pcl::console::find_argument(argc, argv, "-m") >= 0) {
		setUnseenToMaxRange = true;
		std::cout << "Set unseen to max range " << setUnseenToMaxRange << std::endl;
	}
	int temp_coordinate_frame;
	if (pcl::console::parse(argc, argv, "-c", temp_coordinate_frame) >= 0) {
		coordinate_frame = pcl::RangeImage::CoordinateFrame(temp_coordinate_frame);
		std::cout << "Set coordinate frame " << temp_coordinate_frame << std::endl;
	}
	if (pcl::console::parse(argc, argv ,"-r" ,angular_resolution) >= 0) {
		std::cout << "Set angular resolution to " << angular_resolution << std::endl;
	}
	angular_resolution = pcl::deg2rad(angular_resolution);

	// read pcd file
	pcl::PointCloud<PointType>::Ptr point_cloud_ptr(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
	pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;
	Eigen::Affine3f scene_sensor_pose(Eigen::Affine3f::Identity());
	std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument(argc, argv, "pcd");
	if (!pcd_filename_indices.empty()) {
		std::string filename = argv[pcd_filename_indices[0]];
		if (pcl::io::loadPCDFile(filename ,point_cloud) == -1) {
			std::cout << "Was not able to open file." << std::endl;
			printHelp(argv[0]);
			return -1;
		}
		scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(
			point_cloud.sensor_origin_[0],
			point_cloud.sensor_origin_[1],
			point_cloud.sensor_origin_[2]) *
			Eigen::Affine3f(point_cloud.sensor_orientation_)
		);
		std::string far_range_filename = pcl::getFilenameWithoutExtension(filename) + "_far_ranges.pcd";
		if (pcl::io::loadPCDFile(far_range_filename.c_str() ,far_ranges) == -1) {
			std::cout << "Far range file \n" << far_range_filename  << " do not exits." << std::endl;
		}
	}
	else {
		for (float y = -0.5f; y < 0.5f; y+= 0.01f) {
			for (float x = -0.5f; x < 0.5f; x += 0.01f) {
				PointType point;
				point.x = x;
				point.y = y;
				point.z = 2.0f - y;
				point_cloud.points.push_back(point);
			}
		}
		point_cloud.width = point_cloud.size();
		point_cloud.height = 1;
	}
	
	// create range image


	return 0;
}