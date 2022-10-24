#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

	/*
		Advanced Initialization
	*/
	
	// comman intializer
	RowVectorXf vec_4_1(3);
	vec_4_1 << 1, 2, 3;
	RowVectorXf vec_4_2(4);
	vec_4_2 << 1, 9, 8 ,4;
	RowVectorXf vec_4_3(7);
	vec_4_3 << vec_4_1, vec_4_2;
	cout << "vector :" << endl << vec_4_3 << endl << endl;

	MatrixXd mat_4_1(2, 2), mat_4_2(4 ,4);
	mat_4_1 << 1, 2, 3, 4;
	mat_4_2 << mat_4_1, mat_4_1 / 10, mat_4_1 * 10, mat_4_1 * 100;
	cout << "matrix :" << endl << mat_4_2 << endl << endl;

	Matrix3f mat_4_3;
	mat_4_3.row(0) << 1, 2, 3;
	mat_4_3.block(1, 0, 2, 2) << 4, 5, 7, 8;
	mat_4_3.col(2).tail(2) << 6, 9;
	cout << "matrix :" << endl << mat_4_3 << endl << endl;

	// special matrix and vector
	Array33f a_4_4 = Array33f::Zero();
	ArrayXf a_4_5 = ArrayXf::Zero(3);
	ArrayXXf a_4_6 = ArrayXXf::Zero(4,5);

	ArrayXXf table(10,4);
	table.col(0) = ArrayXf::LinSpaced(10 ,0,99);
	table.col(1) = 3.14f / 180 * table.col(0);
	table.col(2) = table.col(0).sin();
	table.col(3) = table.col(0).cos();
	cout << "matrix :" << endl << table << endl << endl;

	const int size = 6;
	MatrixXd mat_4_4(size ,size);
	mat_4_4.topLeftCorner(size / 2, size / 2).setZero();
	mat_4_4.topRightCorner(size / 2, size / 2).setIdentity();
	mat_4_4.bottomLeftCorner(size / 2, size / 2).setIdentity();
	mat_4_4.bottomRightCorner(size / 2, size / 2).setZero();
	cout << "matrix :" << endl << mat_4_4 << endl << endl;

	MatrixXd mat_4_5(size, size);
	mat_4_5 << MatrixXd::Zero(size / 2, size / 2), MatrixXd::Identity(size / 2, size / 2), MatrixXd::Zero(size / 2, size / 2), MatrixXd::Identity(size / 2, size / 2);
	cout << "matrix :" << endl << mat_4_5 << endl << endl;

	// usage as temporary objects to transform matrix
	MatrixXf mat_4_6 = MatrixXf::Random(2,3);
	cout << "matrix :" << endl << mat_4_6 << endl << endl;
	mat_4_6 = (MatrixXf(2,2) << 0 ,1 ,1,0).finished() * mat_4_6;
	cout << "matrix new :" << endl << mat_4_6 << endl << endl;

	/*
		Reduction visitors broadcasting
	*/

	// norm L2 computation
	VectorXf v_6_1(2);
	MatrixXf m_6_1(2, 2), n(2, 2);
	v_6_1 << -1, 3;
	m_6_1 << 1, -2, -3, 4;
	cout << "vector squareNorm " << endl << v_6_1.squaredNorm()  << endl << endl;
	cout << "vector norm " << endl << v_6_1.norm() << endl << endl;
	cout << "vector lpNorm<1>() " << endl << v_6_1.lpNorm<1>() << endl << endl;
	cout << "vector lpNorm<Infinity>() " << endl << v_6_1.lpNorm<Infinity>() << endl << endl;

	cout << "matrix squareNorm " << endl << m_6_1.squaredNorm() << endl << endl;
	cout << "matrix norm " << endl << m_6_1.norm() << endl << endl;
	cout << "matrix lpNorm<1>() " << endl << m_6_1.lpNorm<1>() << endl << endl;
	cout << "matrix lpNorm<Infinity>() " << endl << m_6_1.lpNorm<Infinity>() << endl << endl;

	cout << "self-define 1-norm(mat) " << m_6_1.cwiseAbs().colwise().sum().maxCoeff() << "=="
		<< m_6_1.colwise().lpNorm<1>().maxCoeff() << endl;
	cout << "self-define infty-norm(mat) " << m_6_1.cwiseAbs().rowwise().sum().maxCoeff() << "=="
		<< m_6_1.rowwise().lpNorm<1>().maxCoeff() << endl;

	// boolean reduction
	ArrayXXf a_6_2(2 ,2);
	a_6_2 << 1, 2, 3, 4;
	cout << "all a>0  " << (a_6_2 > 0).all() << endl;
	cout << "any a>0  " << (a_6_2 > 0).any() << endl;
	cout << "count a>0  " << (a_6_2 > 0).count() << endl << endl;
	cout << "all a>2  " << (a_6_2 > 2).all() << endl;
	cout << "any a>2  " << (a_6_2 > 2).any() << endl;
	cout << "count a>2  " << (a_6_2 > 2).count() << endl << endl;

	// visitor
	MatrixXf m_6_2 = MatrixXf::Random(4,8);
	MatrixXf::Index rowIndex, colIndex;
	float max = m_6_2.maxCoeff(&rowIndex , &colIndex);
	cout << "matrix " << endl << m_6_2 << endl;
	cout << "max row " << rowIndex << " col " << colIndex;
	float min = m_6_2.minCoeff(&rowIndex, &colIndex);
	cout << " min row " << rowIndex << " col " << colIndex;

	// partial reduction
	cout << "Column's maximum: " << endl << m_6_2.colwise().maxCoeff() << endl;
	cout << "Row's maximum: " << endl << m_6_2.rowwise().maxCoeff() << endl;

	// partial reduction with other operations
	MatrixXf::Index maxIndex;
	float maxNorm = m_6_2.colwise().sum().maxCoeff(&maxIndex);
	cout << "Column's max line " << maxIndex << endl << m_6_2.col(maxIndex) << endl;

	// broadcasting
	MatrixXf m_6_3(2, 4);
	VectorXf v_6_4(2);
	m_6_3 <<
		1, 2, 3, 4,
		5, 6, 7, 8;
	v_6_4 << -5, 5;
	cout << "matrix " << endl << m_6_3 << endl;
	m_6_3.colwise() += v_6_4;
	cout << "matrix new" << endl << m_6_2 << endl;

	// combining broadcast with operations
	MatrixXf m_6_5(2, 4);
	VectorXf v_6_5(2);
	m_6_5 << 1, 23, 6, 9,
		3, 11, 7, 2;
	v_6_5 << 2, 3;
	MatrixXf::Index combining_broadcast_index;
	(m_6_5.colwise() - v_6_5).colwise().squaredNorm().minCoeff(&combining_broadcast_index);
	cout << "Index " << combining_broadcast_index << endl;

	return 0;
}