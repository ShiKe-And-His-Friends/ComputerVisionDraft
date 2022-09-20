#include<iostream>
#include<pcl/visualization/cloud_viewer.h>
#include<pcl/io/io.h>
#include<pcl/io/pcd_io.h>//pcd��д����ص�ͷ�ļ�
#include<pcl/io/ply_io.h>
#include<pcl/point_types.h> //PCL��֧�ֵĵ�����ͷ�ļ�

int user_data;
using namespace std;


void viewerOneOff(pcl::visualization::PCLVisualizer& viewer) {
	//���ñ�����ɫ
	viewer.setBackgroundColor(1.0, 0.5, 1.0);
}

int plc_new2() {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// char strfilepath[256] = "..\\..\\PCL_Data\\rabbit.pcd";
	char strfilepath[256] = "rabbit.pcd";
	if (-1 == pcl::io::loadPCDFile(strfilepath, *cloud)) {
		cout << "error input!" << endl;
		return -1;
	}

	cout << cloud->points.size() << endl;
	//����viewer����
	pcl::visualization::CloudViewer viewer("Cloud Viewer");  

	viewer.showCloud(cloud);
	viewer.runOnVisualizationThreadOnce(viewerOneOff);
	system("pause");
	return 0;

}