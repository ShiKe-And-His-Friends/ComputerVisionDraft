#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

/*
	Eigen crash in PCL memory.h
		C/C++ -> all configure -> use enhance instruction sets -> arch:/AVX
*/

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 3, "sample cloud");
	viewer->addCoordinateSystem(1.0 ,"global");
	viewer->initCameraParameters();
	return (viewer);
}

int main(int argc, char** argv) {

	std::cout << "sample consensus." << std::endl;

	if (pcl::console::find_argument(argc, argv, "-f") >= 0) {
		std::cout << "-f" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-sf") >= 0) {
		std::cout << "-sf" << std::endl;
	}
	else {
		std::cout << "input" << std::endl;
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 5000;
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(cloud->width * cloud->height);
	for (size_t i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].x = rand() / (RAND_MAX + 1.0);
		cloud->points[i].y = rand() / (RAND_MAX + 1.0);
		if (pcl::console::find_argument(argc, argv, "-s") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0) {
			if (i % 5 == 0) {
				cloud->points[i].z = rand() / (RAND_MAX + 1.0);
			}
			else if (i % 2 == 0) {
				cloud->points[i].z = sqrt(1 - (cloud->points[i].x * cloud->points[i].x) - (cloud->points[i].y * cloud->points[i].y));
			}
			else {
				cloud->points[i].z = - sqrt(1 - (cloud->points[i].x * cloud->points[i].x) - (cloud->points[i].y * cloud->points[i].y));
			}
		}
		else {
			if (i % 2 == 0) {
				cloud->points[i].z = rand() / (RAND_MAX + 1.0);
			}
			else {
				cloud->points[i].z = 1 - (cloud->points[i].x + cloud->points[i].y);
			}
		}
		std::cout << cloud->points[i].x << " " << cloud->points[i].y << " " << cloud->points[i].z << " " << std::endl;
	}
	for (size_t i = 0; i < cloud->points.size(); i++) {
		int radius_distance = 1024;
		cloud->points[i].x = cloud->points[i].x * radius_distance;
		cloud->points[i].y = cloud->points[i].y * radius_distance;
		cloud->points[i].z = cloud->points[i].z * radius_distance;
	}

	// sample consensus model
	std::vector<int> inliers;
	pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(cloud));
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));
	if (pcl::console::find_argument(argc, argv, "-f") >= 0) {
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
		ransac.setDistanceThreshold(0.01);
		ransac.computeModel();
		ransac.getInliers(inliers);
	}
	else if (pcl::console::find_argument(argc, argv, "-sf") >= 0) {
		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_s);
		ransac.setDistanceThreshold(0.01);
		ransac.computeModel();
		ransac.getInliers(inliers);
	}

	// all model inline point to *final
	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final);

	// visualization
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if (pcl::console::find_argument(argc, argv, "-f") >= 0 || pcl::console::find_argument(argc, argv, "-sf") >= 0) {
		viewer = simpleVis(final);
	}
	else {
		viewer = simpleVis(cloud);
	}
	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}
