#include <pcl/common/centroid.h>

// 计算点云均值与标准差
float geyMeanStd(const std::vector<float> &values ,double &mean ,double &stddev);

// 计算两个向量的夹角
float getAngle3D(const Eigen::Vector3f &v1 ,
	const Eigen::Vector3f &v2 ,const bool in_degress);

// 获取点云坐标最值
template <typename PointT> inline void
getMinMax3D(const pcl::PointCloud<PointT> &cloud ,
	PointT &min_pt ,PointT &max_pt);

// 计算指定点到点云最远一点的坐标
template<typename PointT> inline void
getMaxDistance(const pcl::PointCloud<PointT> &cloud ,
	const Eigen::Vector3f &pivot_pt,
	Eigen::Vector3f &max_pt);

//计算点云质心
template <typename PointT, typename Scalar> inline unsigned int
compute3DCentroid(const pcl::PointCloud<PointT> &cloud,
	Eigen::Matrix<Scalar ,4 ,1> &centroid);

// 去点云质心
template <typename PointT, typename Scalar> void
demeanPointCloud(const pcl::PointCloud<PointT>& cloud_in, const Eigen::Matrix<Scalar, 4, 1>& centroid,
	pcl::PointCloud<PointT>& cloud_out);

// 计算点云协方差矩阵
template <typename PointT, typename Scalar> inline unsigned
computeCovarianceMatrix(const pcl::PointCloud<PointT>& cloud,
	const Eigen::Matrix<Scalar, 4, 1>& centroid,
	Eigen::Matrix<Scalar, 3, 3>& covarince_matrix);

// 点云标准化(normalized)协方差矩阵
template <typename PointT, typename Scalar> inline unsigned int
computeMeanAndCovarianceMatrix(const pcl::PointCloud<PointT> &cloud,
	Eigen::Matrix<Scalar ,3, 3> &covariance_matrix,
	Eigen::Matrix<Scalar ,4 ,1> &centroid);

