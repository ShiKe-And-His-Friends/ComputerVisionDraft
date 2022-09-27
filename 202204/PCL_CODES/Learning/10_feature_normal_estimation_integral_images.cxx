/***
	使用积分图进行法线估计
*/
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>

int feature_normal_estimation_integral_images(int agrc, char** argv) {

	std::cout << "feature using integral images" << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("table_scene_mug_stereo_textured.pcd", *cloud);

	// 创建法向量
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	// COVARIANCE_MATRIX  AVERAGE_3D_GRADIENT  AVERAGE_DEPTH——CHANGE
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.02f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setInputCloud(cloud);
	ne.compute(*normals);
	// TODO [20220923] Eigen crash

	// 可视化
	pcl::visualization::PCLVisualizer viewer("PCL viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals);

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}

	return 0;

}