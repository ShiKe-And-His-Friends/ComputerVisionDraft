#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc, char** argv) {

	// matrix resize
	Eigen::MatrixXd x = Eigen::MatrixXd::Random(2, 5);
	cout << "x : " << endl << x << endl;
	x.resize(4, 3);
	cout << "x resize: " << endl << x << endl;
	cout << "x cols: " << x.cols() << " rows:" << x.rows() << endl;

	// vector resize
	Eigen::VectorXd v(4);
	v << 4, 4, 4, 4;
	v.resize(5);
	cout << "v resize: " << endl << v << endl;
	cout << "v cols: " << v.cols() << " rows:" << v.rows() << endl;

	// matrix proprity
	Eigen::MatrixXd mat = Eigen::MatrixXd::Random(6, 6);
	cout << "mat :" << endl << mat << endl << endl;
	cout << "mat 's sum :" << endl << mat.sum() << endl << endl;
	cout << "mat 's prod :" << endl << mat.prod() << endl << endl;
	cout << "mat 's mean :" << endl << mat.mean() << endl << endl;
	cout << "mat 's minCoeff :" << endl << mat.minCoeff() << endl << endl;
	cout << "mat 's maxCoeff :" << endl << mat.maxCoeff() << endl << endl;
	cout << "mat 's trace :" << endl << mat.trace() << endl << endl;

	// Indexs of values
	std::ptrdiff_t i, j;
	float minCoeffs = mat.minCoeff(&i, &j);
	cout << "mat 's minCoeff i " << i << " j " << j << endl;

	// ArrayNt ArrayNNt
	Eigen::ArrayXXd a_2_1(3, 3);
	a_2_1 <<
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	cout << "array abs:" << endl << a_2_1.abs() << endl << endl;
	cout << "array abs sqrt:" << endl << a_2_1.abs().sqrt() << endl << endl;

	// Matrix tranform with ArrayNNt
	Eigen::MatrixXf m = Eigen::MatrixXf::Random(3, 3);
	Eigen::ArrayXXf a = Eigen::ArrayXXf::Random(3, 3);
	cout << "matrix to array:" << endl << m.array() << endl << endl;
	cout << "array to matrix:" << endl << a.matrix() << endl << endl;

	// Block
	cout << "matrix block:" << endl << m.block<2, 2>(1, 1) << endl << endl;
	for (int i = 1; i <= 3; ++i)
	{
		cout << "Block of size " << i << "x" << i << endl;
		cout << m.block(0, 0, i, i) << endl
			<< endl;
	}
		//matrix.block(i,j,p,q) = matrix.block<p,q>(i,j)

	// Block left value
	Eigen::Array22f m_4_1;
	m_4_1 << 1, 2, 3, 4;
	Eigen::Array44f a_4_1 = Eigen::Array44f::Constant(5);
	a_4_1.block<2, 2>(1, 1) = m_4_1;
	cout << "matrix left values: " << endl << a_4_1 << endl << endl;
	a_4_1.block(0,0,2,3) = a_4_1.block(2,1,2,3);
	cout << "matrix right-bottom : " << endl << a_4_1 << endl << endl;

	// Block row and column
	Eigen::MatrixXf m_4_2 = Eigen::MatrixXf::Random(4,4);
	cout << "matrix values: " << endl << m_4_2 << endl << endl;
	cout << "matrix row(0): " << endl << m_4_2.row(0) << endl << endl;
	cout << "matrix col(1): " << endl << m_4_2.col(1) << endl << endl;
	m_4_2.row(3) += m_4_2.row(2);
	cout << "matrix new values: " << endl << m_4_2 << endl << endl;

	// Block corner 
	Eigen::MatrixXf m_4_3 = Eigen::MatrixXf::Random(4, 4);
	cout << "matrix values: " << endl << m_4_3 << endl << endl;
	cout << "matrix leftCols(2): " << endl << m_4_3.leftCols(2) << endl << endl;
	cout << "matrix bottomRows(2): " << endl << m_4_3.bottomRows(2) << endl << endl;
	cout << "matrix topLeftCorner(1,3): " << endl << m_4_3.topLeftCorner(1,3) << endl << endl;
	m_4_3.topLeftCorner(1, 3) = m_4_3.bottomRightCorner(3, 1).transpose();
	cout << "matrix new values: " << endl << m_4_3 << endl << endl;

	// Block of vector
	Eigen::ArrayXf v_4_4(6);
	v_4_4 << 1, 2, 3, 4, 5, 6;
	cout << "vector head(3)" << endl << v_4_4.head(3) << endl << endl;
	cout << "vector tail<3>()" << endl << v_4_4.tail<3>() << endl << endl;
	v_4_4.segment(1, 4) *= 2;
	cout << "vector segment(1, 4) *= 2 :" << endl << v_4_4 << endl << endl;

	return 0;
}