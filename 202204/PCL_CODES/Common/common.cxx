#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointWithScale Scalar;
typedef Eigen::Matrix3f Matrix;
typedef Eigen::Vector3f Vector;

/***

	common.h		��׼C��C++�࣬���к����ĸ���

	angles.h		��׼C�ӿڵĽǶȼ��㺯��

	cnetriod.h		���ĵ�Ĺ��㡢Э�������

	distance.h		��׼C�ӿڵľ�����㺯��

	file_io.h		һЩ�ļ������Ͷ�д�ĺ���

	random.h		������Ƶ����ɺ���
	
	geometry.h		�����ļ��ι��ܵĺ���

	intersection.h	�ߺ����ཻ�ĺ���

	norm.h			��׼C�ӿڼ�����������

	time.h			����ʱ�����ĺ���

	Point_types.h	��������PCLʵ�ֵĵ��Ƶ����ݽṹ����

**/

float rad2deg(float alpha); //���ȵ��Ƕ�

float deg2rad(float aipha); //�Ƕȵ�����

float normaAngle(float alpha); //���򻯽Ƕ�(-PI ,PI)֮��

// ����һȺ3D���ĵ㣬����һ����ά����
Eigen::Vector3f compute3DCentroid(const pcl::PointCloud<PointT> &cloud ,
	Eigen::Matrix<Scalar ,4 ,1>& centroid);

// ����һ����ָ������һ���DnD���Ĺ���
void demeanPointCloud(const pcl::PointCloud<PointT> &cloud_in ,const Eigen::Matrix<Scalar ,4 ,1> &centroid ,pcl::PointCloud <PointT> &cloud );
void computrNDCentroid(const pcl::PointCloud<PointT> &cloud ,Eigen::Matrix<Scalar ,Eigen::Dynamic ,1> &centroid);

// ����������֮��ĽǶ�
float getAngle3D(const Eigen::Vector4f &v1 ,const Eigen::Vector4f &v2 ,const bool in_degresss=false);

// �������ͬ���������ݵľ�ֵ�ͱ�׼��
float getMeanStd(const std::vector<float> &values ,double &mean ,double &stddev);

// ���߽������£���ȡһ����еĵ�
float getPointInBox(const pcl::PointCloud<PointT> &cloud ,Eigen::Vector4f &min_pt ,Eigen::Vector4f &max_pt ,std::vector<int> &indices);

// ����ֱ��ͼ����ֵ
float getMinMax(const PointT &histogram ,int len ,float &min_p ,float &max_p);

// �������ε��Ƶ����
float calculatePolygonArea(const pcl::PointCloud<PointT> &polygon);

// Point_in��ֵ��Point_out
void copyPoint(const PointT &point_in ,PointT &point_out);

// ��ȡ������άֱ��֮��������ά�߶�
void lineToLineSegement(const Eigen::VectorXf &line_a ,const Eigen::VectorXf &line_b ,
	Eigen::Vector4f &pt1_seg ,Eigen::Vector4f &pt2_seg);

// ��õ㵽�ߵ�ƽ������ 
float  sqrPointToLineDistance(const Eigen::Vector4f& pt, const Eigen::Vector4f& line_pt, const Eigen::Vector4f& line_dir);

// ��ȡһ����е����ֶΣ���������С������
void getMaxSegement(const pcl::PointCloud<PointT>& cloud, PointT& pmin, PointT& pmax);

// ȷ����С����ֵ�����Ӧ����������
void eigen22(const Matrix &mat ,Scalar &eigenvalue ,Vector &eigenvector);

// ȷ���Գư�������������������ֵ��Ӧ����������
void computeCorrespondingeigenVector(const Matrix &mat ,const typename Scalar &eigenvalue ,Vector &eigenvector);

// ȷ���Գư��������������С����ֵ����������
void eigen33(const Matrix &mat ,typename Scalar &eigenvalue ,Vector &eigen);

// ����2�׷������
void invert2x2(const Matrix &matrix ,Matrix &inverse);

// ����3�׶Գƾ������
void invert3x3SymMatrix(const Matrix &matrix ,Matrix &inverse);

// ����3�׷��������ʽ
void determinant3x3Matrix(const Matrix &matrix);

// ���Ψһ��3D��ת����Z����ת(0 ,0 ,1)��Y����ת(0 ,1, 0),������������������
void geyTransfromUnitVectorsZY(const Vector &z_axis ,const Vector direction ,Eigen::Affine3f &transformtation);

// ��originת��Ϊ(0 ,0 ,0)�任����Z����ת��(0 ,0,1)��Y����ת��(0,1,0)
void getTransformattionFromIUnitVectorAndOrigin(Vector y_direction ,Vector &z_axis ,Vector &origns ,Eigen::Affine3f &transformation);

// ��ȡ�任�����ŷ����
void geyEularAngles(const Eigen::Transform<Scalar ,3 ,Eigen::Affine> &t ,Scalar &roll ,Scalar &pitch ,Scalar &yaw);

// ��ȡ����ת���е�XYZ��ŷ����
void getTranslationAndEulerAngles(const Eigen::Transform<Scalar ,3 ,Eigen::Affine> &t ,Scalar &x ,Scalar &y ,Scalar &z ,Scalar &roll ,Scalar &pitch ,Scalar &yaw);

// �Ӹ�����ƽ�ƺ�ŷ���Ǵ�����ת����
void getTransformation(float x, float y ,float z ,float roll ,float pitch ,float yaw);

// �����д����һ���������
template<typename Derived> void saveBinary(const Eigen::MatrixBase<Derived> &matrix ,std::ostream &file); //Derived �豸��

// ���������ж�ȡ����
template<typename Derived> void loadBinary(Eigen::MatrixBase<Derived> const &matrix ,std::istream &file);

// ��ȡָ���ֶε�����
template<typename PCLPointCloud2> void geyFieldIndex(const PCLPointCloud2 &cloud ,const std::string &field_name);

// ��ȡ���������е����п����ֶε��б�
void getFieldsList(const pcl::PointCloud<PointT> &cloud);

// ��ȡ�ض��ֶ��������͵Ĵ�С
void getFieldSize(const int datatype);

// ����pcl::PCLPointCloud2���͵ĵ����ֶ�
template<typename PCLPointCloud2> void concatenatePointCloud(const PCLPointCloud2 &cloud1 ,const PCLPointCloud2 &cloud2 ,PCLPointCloud2 &cloud_out);