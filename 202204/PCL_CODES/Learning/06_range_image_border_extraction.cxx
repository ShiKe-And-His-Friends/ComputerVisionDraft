/*********** ERROR *************
	PCL 1.12.1 makes crash
	range_image_border_widght = pcl::visualization::RangeImageVisualizer::getRangeImageBorderWidget(range_image ,...);

	#include <vtkAutoInit.h>
	VTK_MODULE_INIT(vtkRenderingContextOpenGL2)
 **/
/***

	深度图 提取边界

**/

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/file_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <Eigen>
#include <boost/thread.hpp>
#include <pcl/console/parse.h>

namespace Range_image_border_extract {

	typedef pcl::PointXYZ PointType;

	// paramenter
	float angular_resolution = 0.5f;
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
	bool setUnseenToMaxRange = false;

	void printHelp(char *argv[]) {
	
		std::cout
			<< "\nUsage:" << argv[0] << " [options] scene.pcd"
			<< "\t Options:" << "\n"
			<< "\t -r <float> angular resolution in degress (default " << angular_resolution << ") \n"
			<< "\t -c <int> coordinate frame (default " <<(int)coordinate_frame  << ") \n"
			<< "\t -m Treat all unseen points to max range" << "\n"
			<< "\t -h this help" << "\n"
			<< "\t " << "\n"
			<< std::endl;
	}

};

using namespace Range_image_border_extract;


int show_range_border_extraction(int argc ,char **argv) {

	std::cout << "visualization range images" << std::endl;

	if (pcl::console::find_argument(argc ,argv ,"-h") >= 0) {
		printHelp(argv);
		return 0;
	}

	if (pcl::console::find_argument(argc ,argv ,"-m") >=0) {
		setUnseenToMaxRange = true;
		std::cout << "Find unseen range image to max values." << std::endl;
	}
	int temp_coordinate_frames;
	if (pcl::console::parse(argc,argv,"-c",temp_coordinate_frames) >= 0) {
		std::cout <<"Set coordinate frame " << temp_coordinate_frames << std::endl;
		coordinate_frame = pcl::RangeImage::CoordinateFrame(temp_coordinate_frames);
	}
	if (pcl::console::parse(argc, argv, "-r", angular_resolution) >= 0) {
		std::cout << "Set angular resolution " << angular_resolution << std::endl;
	}
	angular_resolution = pcl::deg2rad(angular_resolution);
	std::cout << "Using coordinate frame " << angular_resolution << "deg." << std::endl;

	// read point cloud file
	pcl::PointCloud<PointType>::Ptr point_cloud_ptr(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
	pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;
	Eigen::Affine3f scene_sensor_pose(Eigen::Affine3f::Identity());
	std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument(argc ,argv ,"pcd");
	if (!pcd_filename_indices.empty()) {
		std::string filename = argv[pcd_filename_indices[0]];
		if (pcl::io::loadPCDFile(filename ,point_cloud) == -1) {
			std::cout << "Load pcd file " << filename << " error." << std::endl;
			printHelp(argv);
			return -2;
		}
		std::cout << "Load pcd file " << filename << std::endl;
		scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(
			point_cloud.sensor_origin_[0],
			point_cloud.sensor_origin_[1],
			point_cloud.sensor_origin_[2]) * 
			Eigen::Affine3f(point_cloud.sensor_orientation_));
		std::string far_ranges_filename = pcl::getFilenameWithoutExtension(filename) + "_far_ranges.pcd";
		std::cout << "Load range pcd file " << far_ranges_filename << std::endl;
		if (pcl::io::loadPCDFile(far_ranges_filename.c_str() ,far_ranges) == -1) {
			std::cout << "Load far range file " << far_ranges_filename << " failure." << std::endl;
		}
	}
	else {
		std::cout << "Not input pcd file." << std::endl;
		for (float x = -0.5f; x <= 0.5f; x+= 0.01f) {
			for (float y = -0.5f; y < 0.5f;y += 0.01f) {
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

	// create range image from point cloud
	float noise_level = 0.0;
	float min_range = 0.0f;
	int border_size = 1;
	std::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
	pcl::RangeImage& range_image = *range_image_ptr;
	range_image.createFromPointCloud(point_cloud ,angular_resolution ,pcl::deg2rad(360.0f) ,pcl::deg2rad(180.0f),
		scene_sensor_pose ,coordinate_frame ,noise_level,min_range ,border_size);
	range_image.integrateFarRanges(far_ranges);
	if (setUnseenToMaxRange) {
		range_image.setUnseenToMaxRange();
	}

	// open 3D viewer and add point cloud
	pcl::visualization::PCLVisualizer viewer("3D viewer");
	viewer.setBackgroundColor(1,1,1);
	viewer.addCoordinateSystem(1.0f);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> point_cloud_color_handler(point_cloud_ptr, 0 , 0 , 0);
	viewer.addPointCloud(point_cloud_ptr ,point_cloud_color_handler ,"original point");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler(range_image_ptr, 155, 155, 155);
	viewer.addPointCloud(range_image_ptr ,range_image_color_handler ,"range image");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,2 ,"range image");

	// extract borders
	pcl::RangeImageBorderExtractor border_extractor(&range_image);
	pcl::PointCloud<pcl::BorderDescription> border_descriptions;
	border_extractor.compute(border_descriptions);

	// show in 3D viewer
	pcl::PointCloud<pcl::PointWithRange>::Ptr border_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
	pcl::PointCloud<pcl::PointWithRange>::Ptr veil_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
	pcl::PointCloud<pcl::PointWithRange>::Ptr shadow_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
	pcl::PointCloud<pcl::PointWithRange>& border_points = *border_points_ptr;
	pcl::PointCloud<pcl::PointWithRange>& veil_points = *veil_points_ptr;
	pcl::PointCloud<pcl::PointWithRange>& shadow_points = *shadow_points_ptr;

	for (size_t y = 0; y < (int)range_image.height; y ++) {
		for (size_t x = 0; x < (int)range_image.width; x++) {
			if (border_descriptions.points[y * range_image.width + x].traits[pcl::BORDER_TRAIT__OBSTACLE_BORDER]) {
				border_points.points.push_back(range_image.points[y * range_image.width + x]);
			}
			if (border_descriptions.points[y * range_image.width + x].traits[pcl::BORDER_TRAIT__VEIL_POINT]) {
				veil_points.points.push_back(range_image.points[y * range_image.width + x]);
			}
			if (border_descriptions.points[y * range_image.width + x ].traits[pcl::BORDER_TRAIT__SHADOW_BORDER]) {
				shadow_points.points.push_back(range_image.points[y * range_image.width + x]);
			}
		}
	}

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> border_points_color_handler(border_points_ptr ,0 , 255,0);
	viewer.addPointCloud<pcl::PointWithRange>(border_points_ptr ,border_points_color_handler ,"border points");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,7 ,"border points");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> veil_points_color_handler(veil_points_ptr, 255, 0, 0);
	viewer.addPointCloud<pcl::PointWithRange>(veil_points_ptr, veil_points_color_handler, "veil points");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "veil points");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> shadow_point_color_handler(shadow_points_ptr ,0 ,0, 255);
	viewer.addPointCloud<pcl::PointWithRange>(shadow_points_ptr ,shadow_point_color_handler ,"shadow points");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,7 ,"shadow points");

	pcl::visualization::RangeImageVisualizer* range_image_borders_widget = NULL;
	range_image_borders_widget = pcl::visualization::RangeImageVisualizer::getRangeImageBordersWidget(
		range_image ,-std::numeric_limits<float>::infinity() , std::numeric_limits<float>::infinity(),
		false ,border_descriptions ,"Range image with borders"
		);

	while (!viewer.wasStopped()) {
		range_image_borders_widget->spinOnce();
		viewer.spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(1000000));
	}

	return 0;
}


