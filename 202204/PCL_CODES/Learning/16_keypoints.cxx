/****

	提取NARF特征点，并用图像和3D显示方式进行可视化

**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <Eigen>
#include <boost/thread/thread.hpp>
#include <pcl/common/file_io.h>
//#include <vtkAutoInit.h>
//VTK_MODULE_INIT(vtkRenderingContextOpenGL2)

typedef pcl::PointXYZ PointType;

float angular_resolution = 0.5f;
float support_size = 0.2f;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
bool rotation_invariant = true;
bool setUnseenToMaxRange = false;

void printHelp(const char* console_dir) {
	std::cout <<
		"Usage: " << console_dir << " [options] <example.pcd>" << " \n"
		<< "-r <float> angular resolution in degress ,default " << angular_resolution << "\n"
		<< "-c <int> coordinate frame ,default " << (int)coordinate_frame << " \n"
		<< "-m Treat all unseen point to max range\n"
		<< "-s <float> support size for the interest points (diameter of the used sphere) ,defalut " <<
		support_size << " \n"
		<< "-o <0/1> switch rotational invariant version of the feature on/off ,default " << (int)rotation_invariant << "\n"
		<< "-h help \n"
		<< std::endl;

}

void setViewerPose(pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose) {
	Eigen::Vector3f pose_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
	Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pose_vector;
	Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);
	viewer.setCameraPosition(pose_vector[0], pose_vector[1], pose_vector[2],
		look_at_vector[0], look_at_vector[1], look_at_vector[2],
		up_vector[0], up_vector[1], up_vector[2]);
}

int narf_keypoint_extract(int argc, char** argv) {

	std::cout << "key points." << std::endl;

	// command
	if (pcl::console::find_argument(argc, argv, "-h") >= 0) {
		printHelp(argv[0]);
		return -1;
	}
	if (pcl::console::find_argument(argc, argv, "-m") >= 0) {
		setUnseenToMaxRange = true;
		std::cout << "Setting unseen values in range images to maximum range reading." << std::endl;
	}
	if (pcl::console::parse(argc, argv, "-o", rotation_invariant) >= 0) {
		std::cout << "Setting rotation invariant feature version " << rotation_invariant << std::endl;
	}
	int temp_coordinate_frame;
	if (pcl::console::parse(argc, argv, "-c", temp_coordinate_frame) >= 0) {
		coordinate_frame = pcl::RangeImage::CoordinateFrame(temp_coordinate_frame);
		std::cout << "Using coordinate frame " << (int)coordinate_frame << std::endl;
	}
	if (pcl::console::parse(argc, argv, "-s", support_size) >= 0) {
		std::cout << "Setting support size to " << support_size << std::endl;
	}
	if (pcl::console::parse(argc, argv, "-r", angular_resolution) >= 0) {
		std::cout << "Setting angular resolution " << angular_resolution << " deg." << std::endl;
	}
	angular_resolution = pcl::deg2rad(angular_resolution);

	// read point cloud
	pcl::PointCloud<PointType>::Ptr point_cloud_ptr(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>& point_cloud = *point_cloud_ptr;
	pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;
	Eigen::Affine3f scene_sensor_pose(Eigen::Affine3f::Identity());
	std::vector<int> pcd_filename_indices = pcl::console::parse_file_extension_argument(argc, argv, "pcd");

	if (!pcd_filename_indices.empty()) {
		std::string filename = argv[pcd_filename_indices[0]];
		if (pcl::io::loadPCDFile(filename, point_cloud) == -1) {
			std::cout << "no pcd point cloud find." << std::endl;
			printHelp(argv[0]);
			return -2;
		}
		scene_sensor_pose = Eigen::Affine3f(Eigen::Translation3f(
			point_cloud.sensor_origin_[0], point_cloud.sensor_origin_[1], point_cloud.sensor_origin_[2]))
			* Eigen::Affine3f(point_cloud.sensor_orientation_);
		std::string far_ranges_filename = pcl::getFilenameWithoutExtension(filename) + "_far_ranges.pcd";
		if (pcl::io::loadPCDFile(far_ranges_filename.c_str(), far_ranges) == -1) {
			std::cout << "Far range file " << far_ranges_filename << " does not exits." << std::endl;
		}
	}
	else {
		setUnseenToMaxRange = true;
		std::cout << "\n Np *.pcd file given Generating example point cloud." << std::endl;
		for (float x = -.5f; x <= 0.5f; x += 0.01f) {
			for (float y = -.5f; y <= 0.5f; y += 0.01f) {
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

	// create range image from cloud point.
	float noise_level = 0.0;
	float min_range = 0.0f;
	int border_size = 1;
	boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
	pcl::RangeImage& range_image = *range_image_ptr;
	range_image.createFromPointCloud(point_cloud, angular_resolution, pcl::deg2rad(360.0f), pcl::deg2rad(180.0f),
		scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
	range_image.integrateFarRanges(far_ranges);
	if (setUnseenToMaxRange) {
		range_image.setUnseenToMaxRange();
	}

	// viewer show point cloud
	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	viewer.setBackgroundColor(1, 1, 1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> point_cloud_color_handler(point_cloud_ptr, 0, 0, 0);
	viewer.addPointCloud(point_cloud_ptr, point_cloud_color_handler, "range image");
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler(range_image_ptr, 0, 0, 0);
	//viewer.addPointCloud(range_image_ptr, range_image_color_handler, "range image");
	viewer.initCameraParameters();
	setViewerPose(viewer, range_image.getTransformationToWorldSystem());

	// show range image 
	pcl::visualization::RangeImageVisualizer range_image_widget("Range image");
	range_image_widget.showRangeImage(range_image);

	// Extract NARF points
	pcl::RangeImageBorderExtractor range_image_border_extrator; //Edge
	pcl::NarfKeypoint narf_keypoint_detector; //Point
	narf_keypoint_detector.setRangeImageBorderExtractor(&range_image_border_extrator);
	narf_keypoint_detector.setRangeImage(&range_image);
	narf_keypoint_detector.getParameters().support_size = support_size;
	pcl::PointCloud<int> keypoint_indices;
	narf_keypoint_detector.compute(keypoint_indices);
	std::cout <<"Found " << keypoint_indices.points.size() << std::endl;

	// show extract NARF points
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoint_ptr;
	keypoints.points.resize(keypoint_indices.points.size());
	for (size_t i = 0; i < keypoint_indices.points.size(); i++) {
		keypoints.points[i].getVector3fMap() = range_image.points[keypoint_indices.points[i]].getVector3fMap();
	}
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler(keypoint_ptr ,0 ,255 ,0);
	viewer.addPointCloud<pcl::PointXYZ>(keypoint_ptr ,keypoints_color_handler ,"keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,7,"keypoints");

	// Extract NARF descriptors for interest points
	std::vector<int> keypoint_indices2;
	keypoint_indices2.resize(keypoint_indices.points.size());
	for (size_t i = 0; i < keypoint_indices.size(); i++) {
		keypoint_indices2[i] = keypoint_indices.points[i];
	}
	pcl::NarfDescriptor narf_descriptor(&range_image ,&keypoint_indices2);
	narf_descriptor.getParameters().support_size = support_size;
	narf_descriptor.getParameters().rotation_invariant = rotation_invariant;
	pcl::PointCloud<pcl::Narf36> narf_descriptors;
	narf_descriptor.compute(narf_descriptors);
	std::cout << "Extracted " <<narf_descriptors.size() << " descriptors for " << keypoint_indices.points.size() << std::endl;

	while (!viewer.wasStopped()) {
		range_image_widget.spinOnce();
		viewer.spinOnce();
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	return 0;
}