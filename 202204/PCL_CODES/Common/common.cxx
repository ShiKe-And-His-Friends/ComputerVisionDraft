#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointWithScale Scalar;
typedef Eigen::Matrix3f Matrix;
typedef Eigen::Vector3f Vector;

/***

	common.h		标准C和C++类，所有函数的父类

	angles.h		标准C接口的角度计算函数

	cnetriod.h		中心点的估算、协方差计算

	distance.h		标准C接口的距离计算函数

	file_io.h		一些文件帮助和读写的函数

	random.h		随机点云的生成函数
	
	geometry.h		基本的几何功能的函数

	intersection.h	线和线相交的函数

	norm.h			标准C接口计算矩阵的正则化

	time.h			定义时间计算的函数

	Point_types.h	定义所有PCL实现的点云的数据结构类型

**/

float rad2deg(float alpha); //弧度到角度

float deg2rad(float aipha); //角度到弧度

float normaAngle(float alpha); //正则化角度(-PI ,PI)之间

// 给定一群3D中心点，返回一个三维向量
Eigen::Vector3f compute3DCentroid(const pcl::PointCloud<PointT> &cloud ,
	Eigen::Matrix<Scalar ,4 ,1>& centroid);

// 利用一组点的指数进行一般的DnD中心估计
void demeanPointCloud(const pcl::PointCloud<PointT> &cloud_in ,const Eigen::Matrix<Scalar ,4 ,1> &centroid ,pcl::PointCloud <PointT> &cloud );
void computrNDCentroid(const pcl::PointCloud<PointT> &cloud ,Eigen::Matrix<Scalar ,Eigen::Dynamic ,1> &centroid);

// 计算两向量之间的角度
float getAngle3D(const Eigen::Vector4f &v1 ,const Eigen::Vector4f &v2 ,const bool in_degresss=false);

// 计算给定同步点云数据的均值和标准差
float getMeanStd(const std::vector<float> &values ,double &mean ,double &stddev);

// 给边界的情况下，获取一组框中的点
float getPointInBox(const pcl::PointCloud<PointT> &cloud ,Eigen::Vector4f &min_pt ,Eigen::Vector4f &max_pt ,std::vector<int> &indices);

// 点云直方图的最值
float getMinMax(const PointT &histogram ,int len ,float &min_p ,float &max_p);

// 计算多边形点云的面积
float calculatePolygonArea(const pcl::PointCloud<PointT> &polygon);

// Point_in赋值给Point_out
void copyPoint(const PointT &point_in ,PointT &point_out);

// 获取两条三维直线之间的最短三维线段
void lineToLineSegement(const Eigen::VectorXf &line_a ,const Eigen::VectorXf &line_b ,
	Eigen::Vector4f &pt1_seg ,Eigen::Vector4f &pt2_seg);

// 获得点到线的平方距离 
float  sqrPointToLineDistance(const Eigen::Vector4f& pt, const Eigen::Vector4f& line_pt, const Eigen::Vector4f& line_dir);

// 获取一组点中的最大分段，并返回最小和最大点
void getMaxSegement(const pcl::PointCloud<PointT>& cloud, PointT& pmin, PointT& pmax);

// 确定最小特征值和其对应的特征向量
void eigen22(const Matrix &mat ,Scalar &eigenvalue ,Vector &eigenvector);

// 确定对称半正定输入矩阵给定特征值对应的特征向量
void computeCorrespondingeigenVector(const Matrix &mat ,const typename Scalar &eigenvalue ,Vector &eigenvector);

// 确定对称半正定输入矩阵最小特征值和特征向量
void eigen33(const Matrix &mat ,typename Scalar &eigenvalue ,Vector &eigen);

// 计算2阶方阵的逆
void invert2x2(const Matrix &matrix ,Matrix &inverse);

// 计算3阶对称矩阵的逆
void invert3x3SymMatrix(const Matrix &matrix ,Matrix &inverse);

// 计算3阶方阵的行列式
void determinant3x3Matrix(const Matrix &matrix);

// 获得唯一的3D旋转，讲Z轴旋转(0 ,0 ,1)，Y轴旋转(0 ,1, 0),并且两个轴是正交的
void geyTransfromUnitVectorsZY(const Vector &z_axis ,const Vector direction ,Eigen::Affine3f &transformtation);

// 讲origin转化为(0 ,0 ,0)变换，将Z轴旋转成(0 ,0,1)，Y轴旋转成(0,1,0)
void getTransformattionFromIUnitVectorAndOrigin(Vector y_direction ,Vector &z_axis ,Vector &origns ,Eigen::Affine3f &transformation);

// 提取变换矩阵的欧拉角
void geyEularAngles(const Eigen::Transform<Scalar ,3 ,Eigen::Affine> &t ,Scalar &roll ,Scalar &pitch ,Scalar &yaw);

// 提取给定转换中的XYZ与欧拉角
void getTranslationAndEulerAngles(const Eigen::Transform<Scalar ,3 ,Eigen::Affine> &t ,Scalar &x ,Scalar &y ,Scalar &z ,Scalar &roll ,Scalar &pitch ,Scalar &yaw);

// 从给定的平移和欧拉角创建旋转矩阵
void getTransformation(float x, float y ,float z ,float roll ,float pitch ,float yaw);

// 保存或写矩阵到一个输出流中
template<typename Derived> void saveBinary(const Eigen::MatrixBase<Derived> &matrix ,std::ostream &file); //Derived 设备流

// 从输入流中读取矩阵
template<typename Derived> void loadBinary(Eigen::MatrixBase<Derived> const &matrix ,std::istream &file);

// 获取指定字段的索引
template<typename PCLPointCloud2> void geyFieldIndex(const PCLPointCloud2 &cloud ,const std::string &field_name);

// 获取给定点云中的所有可用字段的列表
void getFieldsList(const pcl::PointCloud<PointT> &cloud);

// 获取特定字段数据类型的大小
void getFieldSize(const int datatype);

// 链接pcl::PCLPointCloud2类型的点云字段
template<typename PCLPointCloud2> void concatenatePointCloud(const PCLPointCloud2 &cloud1 ,const PCLPointCloud2 &cloud2 ,PCLPointCloud2 &cloud_out);