/***

	��ƥ��������

	Pairwise incremental �ɶ�����

**/
#include <Eigen>
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_representation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

using pcl::visualization::PointCloudColorHandlerCustom;
using pcl::visualization::PointCloudColorHandlerGenericField;

pcl::visualization::PCLVisualizer* p;
int vp_1(0), vp_2(1);

// ����һ���ṹ�巽��Ե����ļ��͵��ƶ�����гɶԴ���͹������
struct PCD {

	PointCloud::Ptr cloud; //���ƹ���ָ��
	std::string f_name; // �ļ���

	PCD() : cloud(new PointCloud) {};

};

// �¶���� <x ,y ,z ,curvature>
class MyPointRepresentation : public pcl::PointRepresentation<PointNormalT> {
	using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;

public:
	MyPointRepresentation() {
		nr_dimensions_ = 4; // ����ά��
	}

	virtual void copyToFloatArray(const PointNormalT& p, float* out) const {
		out[0] = p.x;
		out[1] = p.y;
		out[2] = p.z;
		out[3] = p.curvature;
	}

};

void loadData(int argc, char** argv, std::vector<PCD, Eigen::aligned_allocator<PCD>>& models) {
	std::string extension(".pcd");
	for (int i = 0; i < argc; i++) {
		std::string fname = std::string(argv[i]);
		if (fname.size() <= extension.size()) {
			continue;
		}
		std::transform(fname.begin(), fname.end(), fname.begin(), (int (*)(int))tolower);
		// �ж�pcd��׺
		if (fname.compare(fname.size() - extension.size(), extension.size(), extension) == 0) {
			PCD m;
			m.f_name = argv[i];
			pcl::io::loadPCDFile(argv[i], *m.cloud);
			// �ӵ����Ƴ�NAN
			std::vector<int> indices;
			pcl::removeNaNFromPointCloud(*m.cloud, *m.cloud, indices);
			models.push_back(m);
		}
	}

}

void showCloudLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source) {
	p->removePointCloud("vp1_target");
	p->removePointCloud("vp1_source");

	PointCloudColorHandlerCustom<PointT> tgt_h(cloud_target, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> src_h(cloud_source, 255, 0, 0);
	p->addPointCloud(cloud_target, tgt_h, "vp1_target", vp_1);
	p->addPointCloud(cloud_source, src_h, "vp1_source", vp_1);

	PCL_INFO("Press q to begin the registration.\n");
	p->spin();
}

void showCloudRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source) {
	p->removePointCloud("target");
	p->removePointCloud("source");

	PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler(cloud_target, "curature");
	PointCloudColorHandlerGenericField<PointNormalT> src_color_handler(cloud_source, "curature");
	if (!tgt_color_handler.isCapable()) {
		PCL_WARN("Cannot create curvature color handler");
	}
	if (!src_color_handler.isCapable()) {
		PCL_WARN("Cannot create curvature color handler");
	}

	p->addPointCloud(cloud_target, tgt_color_handler, "target", vp_2);
	p->addPointCloud(cloud_source, src_color_handler, "source", vp_2);
	p->spinOnce();
}

//ʵ��ƥ��,���в���������һ����Ҫƥ��ĵ��ƣ��Լ��Ƿ�����²������������������׼��ĵ���
void pairAlign(const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f& final_transform, bool downsample = false) {
	PointCloud::Ptr src(new PointCloud);
	PointCloud::Ptr tgt(new PointCloud);
	pcl::VoxelGrid<PointT> grid;
	if (downsample) {
		grid.setLeafSize(0.05, 0.05, 0.05);
		grid.setInputCloud(cloud_src);
		grid.filter(*src);
		grid.setInputCloud(cloud_tgt);
		grid.filter(*tgt);
	}
	else {
		src = cloud_src;
		tgt = cloud_tgt;
	}

	// ����������ʺͷ�����
	PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
	PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);
	pcl::NormalEstimation<PointT, PointNormalT> norm_set;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	norm_set.setSearchMethod(tree);
	norm_set.setKSearch(30);
	norm_set.setInputCloud(src);
	norm_set.compute(*points_with_normals_src);
	std::cout << "Source Points Number " << (*points_with_normals_src).points.size() << std::endl;
	pcl::copyPointCloud(*src, *points_with_normals_src);

	norm_set.setInputCloud(tgt);
	norm_set.compute(*points_with_normals_tgt);
	std::cout << "Target Points Number " << (*points_with_normals_tgt).points.size() << std::endl;
	pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

	// Instantiate our custom point representation and weight the 'curvature' dimension
	MyPointRepresentation point_representation;
	float alpha[4] = { 1.0 ,1.0 ,1.0 ,1.0 };
	point_representation.setRescaleValues(alpha);

	// ��׼
	pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
	reg.setTransformationEpsilon(1e-6); // ��������������ԽС����Խ������Խ��
	reg.setMaxCorrespondenceDistance(0.1);
	//reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));
	reg.setPointRepresentation(std::make_shared<const MyPointRepresentation>(point_representation));
	reg.setInputSource(points_with_normals_src);
	reg.setInputTarget(points_with_normals_tgt);

	Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
	PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
	reg.setMaximumIterations(2);
	for (int i = 0; i < 30; i++) {
		PCL_INFO("Iteration Nr. %d \n", i);
		points_with_normals_src = reg_result;
		// Estimate
		reg.setInputSource(points_with_normals_src);
		reg.align(*reg_result);
		// accumulate transformation between each Iteration
		Ti = reg.getFinalTransformation() * Ti;
		// if different between this transformation an the previous one 
		// is samller than the threshold , refine the process by reducing
		// the maximal correspondence distance
		if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon()) {
			reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);
		}
		prev = reg.getLastIncrementalTransformation();
		showCloudRight(points_with_normals_tgt, points_with_normals_src);
	}
	targetToSource = Ti.inverse();

	pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);
	p->removePointCloud("source");
	p->removePointCloud("target");

	PointCloudColorHandlerCustom<PointT> cloud_tgt_h(output, 0, 255, 0);
	PointCloudColorHandlerCustom<PointT> cloud_src_h(cloud_src, 255, 0, 0);
	p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
	p->addPointCloud(cloud_src, cloud_src_h, "source", vp_2);
	PCL_INFO("Press q to continue the registration");
	p->spin();
	p->removePointCloud("source");
	p->removePointCloud("target");
	*output += *cloud_src;
	final_transform = targetToSource;
}

// input pcd : capture0001.pcd  capture0002.pcd capture0003.pcd capture0004.pcd  capture0005.pcd  frame_00000.pcd
int registration_pairwise_incremental(int argc, char** argv) {

	std::cout << "registration pairwise incremental." << std::endl;

	// �������е���
	std::vector<PCD, Eigen::aligned_allocator<PCD>> data;
	loadData(argc, argv, data);
	if (data.empty()) {
		PCL_ERROR("Synatx is:%s <source.pcd> <target.pcd> [*]", argv[0]);
		PCL_ERROR("[*] - multiple files cna be added. The registration results of (i,i+1) will be registered against (i+2) ,etc");
		return -1;
	}
	PCL_INFO("Loaded %d datasets.", (int)data.size());

	p = new pcl::visualization::PCLVisualizer(argc, argv, "Pairwise Incremental Registration example.");
	p->createViewPort(0.0, 0.0, 0.5, 1.0, vp_1);
	p->createViewPort(0.5, 0.0, 1.0, 1.0, vp_2);
	PointCloud::Ptr result(new PointCloud), source, target;
	Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;
	for (size_t i = 1; i < data.size(); i++) {
		source = data[i - 1].cloud;
		target = data[i].cloud;
		showCloudLeft(source, target);

		PointCloud::Ptr temp(new PointCloud);
		PCL_INFO("Aligning %s (%d) with %s (%d).\n", data[i - 1].f_name.c_str(), source->points.size(), data[i].f_name.c_str(), target->points.size());

		// pairTransform ���ش�Ŀ�����target��source�ı任����
		pairAlign(source, target, temp, pairTransform, true);
		//�ѵ�ǰ������׼��ĵ���tempת����ȫ������ϵ����result
		pcl::transformPointCloud(*temp, *result, GlobalTransform);

		GlobalTransform = GlobalTransform * pairTransform;

		std::stringstream ss;
		ss << i << ".pcd";
		pcl::io::savePCDFile(ss.str(), *result, true);
	}

	return 0;
}