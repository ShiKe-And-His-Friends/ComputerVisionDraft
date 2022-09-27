/***
	PFH 点特征直方图 估计和描述特征
	Point Feature Histogram
*/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/features/pfh.h>
#include <boost/thread.hpp>

#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingContextOpenGL2)

int feature_point_histogram(int argc, char** argv) {

	std::cout << "feature PFH" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("ism_test_cat.pcd" ,*cloud);
	pcl::PointCloud < pcl::Normal>::Ptr normals(new pcl::PointCloud < pcl::Normal > );
	normals->points.resize(cloud->points.size());

	pcl::PFHEstimation < pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125 > pfh;
	pfh.setInputCloud(cloud);
	pfh.setInputNormals(normals);
	// 空的kd树传给PFH评估对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	// pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	pfh.setSearchMethod(tree); //输出数据集
	
	pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs(new pcl::PointCloud<pcl::PFHSignature125>);
	pfh.setRadiusSearch(0.05);//半径5厘米
	pfh.compute(*pfhs);

	// pfhs->points.size() == input_cloud->points.size()
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*pfhs, 300);
	plotter.plot();

	return 0;
}