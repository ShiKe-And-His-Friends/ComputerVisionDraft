#include <iostream>
#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>

int main(int argc ,char **argv) {
	std::cout << "range image create" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloudPtr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>& pointCloud = *pointCloudPtr;

	// create matirx point cloud
	for (float y = -0.5f; y <= 0.5f; y+=0.01f) {
		for (float z = -0.5f; z <= 0.5f; z+=0.01f) {
			pcl::PointXYZ point;
			point.x = 2.0f - y;
			point.y = y;
			point.z = z;
			pointCloud.points.push_back(point);
		}
	}

	float angularResolution = (float)(1.0f * (M_PI / 180.0f)); //angele 1 rad
	float maxAngleWidth = (float)(360.0f * (M_PI / 180.0f));
	float maxAngleHeight = (float)(360.0f * (M_PI / 180.0f));
	Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f ,0.0f ,0.0f);
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
	float noiseLevel = 0.00;
	float minRange = 0.0f;
	int borderSize = 1;

	boost::shared_ptr<pcl::RangeImage> range_image_ptr(new pcl::RangeImage);
	pcl::RangeImage &rangeImage = *range_image_ptr;
	
	//pcl::PointCloud<pcl::RangeImage>::Ptr range_image_ptr(new pcl::PointCloud<pcl::RangeImage>);
	//pcl::RangeImage& rangeImage = (*range_image_ptr)[0];

	rangeImage.createFromPointCloud(pointCloud ,angularResolution ,maxAngleWidth ,maxAngleHeight,
			sensorPose ,coordinate_frame ,noiseLevel ,minRange ,borderSize);
	std::cout << rangeImage << "\n" << std::endl;

	// viewer
	pcl::visualization::PCLVisualizer viewer("3D viewer");
	viewer.setBackgroundColor(1 ,1,1);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::RangeImage> range_image_color_handler(range_image_ptr, 0, 0, 0);
	viewer.addPointCloud(range_image_ptr ,range_image_color_handler ,"range image");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,3 , "range image");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> org_image_color_handler(pointCloudPtr ,255 ,0 ,0);
	viewer.addPointCloud(pointCloudPtr , org_image_color_handler ,"orginal image");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,5 ,"orginal image");
	
	viewer.initCameraParameters();
	viewer.addCoordinateSystem(1.0);
	while (!viewer.wasStopped()) {
		viewer.spinOnce();
		//pcl_sleep(0.01);
	}

	return 0;
}