/***
	
	点云匹配 交互式ICP

	ICP Iteractive Closest Points

**/
#include <iostream>
#include <Eigen>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false;

void print4x4Matrix(const Eigen::Matrix4d & matrix) {
	std::cout << "Rotation matrix:" << std::endl;
	printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n" ,matrix(0 ,0) ,matrix(0,1) ,matrix(0,2));
	printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("\t|%6.3f\t%6.3f\t%6.3f\t|\n", matrix(2, 0), matrix(2, 1), matrix(2, 2));

	std::cout << "Translation vector:" << std::endl;	
	printf("\tt=<%6.3f\t%6.3f\t%6.3f\t>\n", matrix(0,3), matrix(1, 3), matrix(2, 3));
}

// 查看窗口的回调函数，当查看器处于顶部，按任意键会回调该方法
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing) {
	if (event.getKeySym() == "space" && event.keyDown()) {
		next_iteration = true;
	}
}

int registration_interactive_monkey(int argc ,char ** argv) {

	std::cout << "registration interactive icp" << std::endl;

	PointCloudT::Ptr cloud_in(new PointCloudT);
	PointCloudT::Ptr cloud_tr(new PointCloudT);
	PointCloudT::Ptr cloud_icp(new PointCloudT);

	if (argc < 2) {
		std::cout << "Usage:" << std::endl;
		printf("\t%s file.ply number_of_ICP iteration\n", argv[0]);
		PCL_ERROR("Provide one ply file.\n");
		return -1;
	}
	int iterations = 1;
	if (argc > 2) {
		iterations = atoi(argv[2]);
		if (iterations < 1) {
			iterations = 1;
		}
	} 

	pcl::console::TicToc time;
	time.tic();
	if (pcl::io::loadPLYFile(argv[1],*cloud_in) < 0) {
		PCL_ERROR("Error loading cloud %s.\n",argv[1]);
		return -1;
	}
	std::cout << "points " << cloud_in->size()  << std::endl;

	// 刚性变换改变原始点云
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
	double theta = M_PI / 8;
	transformation_matrix(0, 0) = std::cos(theta);
	transformation_matrix(0, 1) = -sin(theta);
	transformation_matrix(1, 0) = sin(theta);
	transformation_matrix(1, 1) = std::cos(theta);
	// 设置Z轴平移
	transformation_matrix(2, 3) = 0.4;
	print4x4Matrix(transformation_matrix);

	//执行刚性变换
	pcl::transformPointCloud(*cloud_in ,*cloud_icp ,transformation_matrix);
	*cloud_tr = *cloud_icp;

	//创建ICP对象
	time.tic();
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setMaximumIterations(iterations);
	icp.setInputSource(cloud_icp);
	icp.setInputTarget(cloud_in);
	icp.align(*cloud_icp);
	icp.setMaximumIterations(1); //下次调用align的参数
	std::cout << "Applied " << iterations << " ICP iteration(s) in " << time.toc() << std::endl;
	// 检查ICP是否收敛
	if (icp.hasConverged()) {
		transformation_matrix = icp.getFinalTransformation().cast<double>();
		std::cout << "ICP converge matrix:" << std::endl;
		print4x4Matrix(transformation_matrix);
	}
	else {
		PCL_ERROR("ICP has not converged.");
		return -2;
	}

	// 显示visualization
	pcl::visualization::PCLVisualizer viewer("ICP demo");
	int v1(0), v2(1);
	viewer.createViewPort(0, 0 , .5 ,1 ,v1);
	viewer.createViewPort(.5 , 0 ,1, 1, v2);
	float bckgr_gray_level = 0.0; //black
	float txt_gray_lvl = 1.0 - bckgr_gray_level;

	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in , (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl, (int)255 * txt_gray_lvl);
	viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
	viewer.addPointCloud(cloud_in ,cloud_in_color_h ,"cloud_in_v2" ,v2);
	// 旋转的点云为绿色
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h(cloud_tr, 20, 180, 20);
	viewer.addPointCloud(cloud_tr, cloud_tr_color_h, "color_tr_v1", v1);
	// align对齐点云设置红色
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(cloud_icp, 180, 20, 20);
	viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "color_icp_v2", v2);
	viewer.addText("White: original point cloud Green:",10 ,15,16 ,txt_gray_lvl ,txt_gray_lvl ,txt_gray_lvl ,"icp_info_1" ,v1);
	viewer.addText("White: ICP align point cloud Red:", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
	viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);
	viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0); // position and orientation
	viewer.setSize(1280 ,1024);
	viewer.registerKeyboardCallback(&keyboardEventOccurred ,(void*)NULL);

	std::stringstream ss;
	ss << iterations;
	std::string iterations_cnt = "ICP iterations = " + ss.str();
	viewer.addText(iterations_cnt ,10 ,60 ,16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl ,"iterations_cnt" ,v2);

	while (!viewer.wasStopped()) {
		viewer.spinOnce();
		if (next_iteration) {
			time.tic();
			
			icp.align(*cloud_icp);
			std::cout << "Applied 1 ICP iteration in " << time.toc() << " ms " << std::endl;
			if (icp.hasConverged()) {
				// C语言小技巧：增加11行覆盖最后一个矩阵
				printf("\033[11A");
				printf("\nICP has converged ,score is %+.0e\n" ,icp.getFitnessScore());
				std::cout << "\nICP transformation  iteration threshold" << ++iterations << std::endl;
				transformation_matrix *= icp.getFinalTransformation().cast<double>();
				print4x4Matrix(transformation_matrix);

				ss.str("");
				ss << iterations;
				std::string iterations_cnt = "ICP iterations = " + ss.str();
				viewer.updateText(iterations_cnt ,10 ,60 ,16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl ,"iterations_cnt");
				viewer.updatePointCloud(cloud_icp ,cloud_icp_color_h ,"cloud_icp_v2");
			}
			else {
				PCL_ERROR("\nICP has not converged.\n");
				return -1;
			}
		}
		next_iteration = false;
	}

	return 0;
}