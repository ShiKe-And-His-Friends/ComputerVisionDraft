/***

	ʶ��  ����ģ�� ISM Imolicit Shape Model

	��ģ��ʶ�����ƴʴ�ģ�ͣ��������е��Ƶ������㡢K��ֵ���࣬�õ��Ӿ��ʵ䡣����ȷ��һ��ģ��Ԥ����������ġ�
	ISM�㷨�����Ǵ�ţBastian Leibe��д��ISM�㷨�ĺü������£��Ϻõ��������˺ͳ���ʶ�� Full-Resoliution Residual Networks for Semantic in Street Scenes / CVPR-12

	ISM�㷨����������֣�ѵ��������ʶ��
	1. ���ؼ��� voxel grid
	2. �����������������ֱ��ͼFPFH�����㷨������
	3. ��K-means�����㷨���������Ӿ��ʵ�
	4. ����ÿһ��ʵ����һ�������أ�һ���Ӿ��ʵ䣩�������ؼ���
	5. ��ÿһ���Ӿ��ʵ䣬���ݹؼ�������ĵķ���������ͳ��Ȩ��
	6. ��ÿһ���ؼ��㣬ѧϰȨ�أ����㵽�������ľ���

	����ʶ��Ҳ���Ƕ��������Ĺ��̣�
	1. ��������
	2. ��ÿ���������������������
	3. ��ÿ��������������������������ӣ���ѵ���׶ε������ʵ�
	4. ��ÿһ����������㣬 ����voteȨ��
	5. ��������������Ϊһ�����򼯣�Ԥ�����ĺ�����

	./implicit_shape_model.exe 
		ism_train_cat.pcd 0
		ism_train_horse.pcd 1
		ism_train_lioness.pcd 2
		ism_train_michael.pcd 3
		ism_train_wolf.pcd 4
		is_test_cat.pcd 0

**/

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/recognition/implicit_shape_model.h>
#include <pcl/recognition/impl/implicit_shape_model.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc ,char ** argv) {

	std::cout << "Recognition implicit shape model." << std::endl;

	if (argc == 0 || argc %2 == 0) {
		std::cout << "input arguments not match." << std::endl;
		return (-1);
	}
	unsigned int number_of_training_clouds = (argc - 3) / 2;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
	normal_estimator.setRadiusSearch(2.50f);

	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> training_clouds;
	std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;
	std::vector<unsigned int> training_classes; // ѵ�����Ƶ����

	// ��ȡ��������
	for (size_t i_cloud = 0; i_cloud < number_of_training_clouds - 1; i_cloud++) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr tr_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[i_cloud *2 + 1] ,*tr_cloud) == -1) {
			std::cout << "load pcd file failure: " << argv[i_cloud*2 +1] << std::endl;
			return -2;
		}
		pcl::PointCloud<pcl::Normal>::Ptr tr_normals = (new pcl::PointCloud<pcl::Normal>)->makeShared();
		normal_estimator.setInputCloud(tr_cloud);
		normal_estimator.compute(*tr_normals);

		unsigned int tr_class = static_cast<unsigned int>(strtol(argv[i_cloud*2+2] ,0 ,10)); //strtol str->long
		training_clouds.push_back(tr_cloud);
		training_normals.push_back(tr_normals);
		training_classes.push_back(tr_class);
	}

	// �÷���������ȡFPHF��������ֱ��ͼ ���㷨�߼н�
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>>::Ptr fpfh (new pcl::FPFHEstimation<pcl::PointXYZ ,pcl::Normal ,pcl::Histogram<153>>);
	fpfh->setRadiusSearch(3.0f);
	pcl::Feature < pcl::PointXYZ, pcl::Histogram<153>>::Ptr feature_estimator(fpfh);
	pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal> ism; //����������
	ism.setFeatureEstimator(feature_estimator);
	ism.setTrainingClouds(training_clouds);
	ism.setTrainingNormals(training_normals);
	ism.setTrainingClasses(training_classes);
	ism.setSamplingSize(2.0f);

	pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>::ISMModelPtr model = std::shared_ptr<pcl::features::ISMModel>(new pcl::features::ISMModel); //ģ��
	ism.trainISM(model); //ѵ��ģ��

	std::string file("train_ism_model.txt");
	model->saveModelToFile(file);
	model->loadModelFromfile(file);

	// ������Ե���
	unsigned int testing_class = static_cast<unsigned int>(strtol(argv[argc - 1], 0, 10));
	pcl::PointCloud<pcl::PointXYZ>::Ptr testing_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[argc - 2], *testing_cloud) == -1) {
		std::cout << "load pcd file failure: " << argv[argc - 2] << std::endl;
		return -4;
	}
	pcl::PointCloud<pcl::Normal>::Ptr testing_normals = (new pcl::PointCloud<pcl::Normal>)->makeShared();
	normal_estimator.setInputCloud(testing_cloud);
	normal_estimator.compute(*testing_normals);

	// Ѱ�ҵ���
	std::shared_ptr<pcl::features::ISMVoteList<pcl::PointXYZ>> vote_list = ism.findObjects(
		model,
		testing_cloud,
		testing_normals,
		testing_class
	);

	// �������������̣��㷨��ȥ��testing_class�������塣���߽����pcl::ism::ISMVoteList��ʽ����
	double radius = model->sigmas_[testing_class] * 10.0;
	double sigma = model->sigmas_[testing_class];
	std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak>> strongest_peaks;
	vote_list->findStrongestPeaks(strongest_peaks ,testing_class ,radius ,sigma);

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = (new pcl::PointCloud<pcl::PointXYZRGB>)->makeShared();
	colored_cloud->height = 0;
	colored_cloud->width = 1;
	pcl::PointXYZRGB point;
	point.r = 255;
	point.g = 255;
	point.b = 255;

	for (size_t i_point = 0; i_point < testing_cloud->points.size();i_point ++) {
		point.x = testing_cloud->points[i_point].x;
		point.y = testing_cloud->points[i_point].y;
		point.z = testing_cloud->points[i_point].z;
		colored_cloud->points.push_back(point);
	}
	colored_cloud->height += testing_cloud->points.size();
	point.r = 255;
	point.g = 0;
	point.b = 0;
	for (size_t i_vote = 0; i_vote < strongest_peaks.size(); i_vote++) {
		point.x = strongest_peaks[i_vote].x;
		point.y = strongest_peaks[i_vote].y;
		point.z = strongest_peaks[i_vote].z;
		colored_cloud->points.push_back(point);
	}
	colored_cloud->height += strongest_peaks.size();

	// ��ͼ�鿴
	pcl::visualization::CloudViewer viewer("Result viewer");
	viewer.showCloud(colored_cloud);
	while (!viewer.wasStopped()) {


	}
	return 0;
}