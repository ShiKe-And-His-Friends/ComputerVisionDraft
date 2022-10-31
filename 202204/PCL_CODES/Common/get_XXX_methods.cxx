#include <pcl/common/centroid.h>

// ������ƾ�ֵ���׼��
float geyMeanStd(const std::vector<float> &values ,double &mean ,double &stddev);

// �������������ļн�
float getAngle3D(const Eigen::Vector3f &v1 ,
	const Eigen::Vector3f &v2 ,const bool in_degress);

// ��ȡ����������ֵ
template <typename PointT> inline void
getMinMax3D(const pcl::PointCloud<PointT> &cloud ,
	PointT &min_pt ,PointT &max_pt);

// ����ָ���㵽������Զһ�������
template<typename PointT> inline void
getMaxDistance(const pcl::PointCloud<PointT> &cloud ,
	const Eigen::Vector3f &pivot_pt,
	Eigen::Vector3f &max_pt);

//�����������
template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroid(const pcl::PointCloud<PointT> &cloud,
	Eigen::Matrix<Scalar ,4 ,1> &centroid);

// ȥ��������
template <typename PointT, typename Scalar> void
demeanPointCloud(const pcl::PointCloud<PointT>& cloud_in, const Eigen::Matrix<Scalar, 4, 1>& centroid,
	pcl::PointCloud<PointT>& cloud_out);

// �������Э�������
template <typename PointT, typename Scalar> inline unsigned
computeCovarianceMatrix(const pcl::PointCloud<PointT>& cloud,
	const Eigen::Matrix<Scalar, 4, 1>& centroid,
	Eigen::Matrix<Scalar, 3, 3>& covarince_matrix);

// ���Ʊ�׼��(normalized)Э�������
template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrix(const pcl::PointCloud<PointT> &cloud,
	Eigen::Matrix<Scalar ,3, 3> &covariance_matrix,
	Eigen::Matrix<Scalar ,4 ,1> &centroid);

