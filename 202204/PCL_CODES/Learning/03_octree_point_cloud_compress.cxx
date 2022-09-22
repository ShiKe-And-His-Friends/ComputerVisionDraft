#include <iostream>
#include <pcl/io/openni2_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/pcl_macros.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <boost/signals2.hpp>

class SimpleOpenNIViewer {

	pcl::visualization::CloudViewer viewer;
	pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudEncoder;
	pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudDecoder;

public:
	SimpleOpenNIViewer() : viewer("point cloud compress viewer") {}

	// OpenNI2 callback methods
	void cloud_cb_(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud) {

		if (!viewer.wasStopped()) {
			// save bytes stream
			std::stringstream compressionData;
			pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZRGBA>());

			// compress
			PointCloudEncoder->encodePointCloud(cloud, compressionData);

			// decompress
			PointCloudDecoder->decodePointCloud(compressionData, cloudOut);

			viewer.showCloud(cloudOut);
		}
	}

	void run() {
		bool showStatistics = true;
		pcl::io::compression_Profiles_e compressionProfile = pcl::io::MED_RES_OFFLINE_COMPRESSION_WITH_COLOR;

		PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>(compressionProfile, showStatistics);
		PointCloudDecoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>();
		
		// openNI2 get each frame data
		pcl::Grabber *interface_gradbber = new pcl::io::OpenNI2Grabber();

		std::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = std::bind(&SimpleOpenNIViewer::cloud_cb_, this ,std::placeholders::_1);
		boost::signals2::connection c = interface_gradbber->registerCallback(f);
		interface_gradbber->start();
		while (!viewer.wasStopped()) {
			boost::this_thread::sleep(boost::posix_time::seconds(1));
		}
		interface_gradbber->stop();

		delete(PointCloudEncoder);
		delete(PointCloudDecoder);
	}

};

int octree_point_cloud_compress(int argc ,char **argv) {
	std::cout << "octree point cloud compress." << std::endl;
	SimpleOpenNIViewer v;
	v.run();

	return 0;
}