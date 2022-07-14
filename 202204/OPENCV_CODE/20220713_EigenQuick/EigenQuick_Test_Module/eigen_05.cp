#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

	/*
		Map class

		Map<Matrix<typename Scalar ,int RowsAtCompilerTime ,int ColsAtCompileTime>>
		Map<const Vector4i> mi(int*)
		Map<typename MatrixType , int MapOptions ,typename StrideType>
	*/
	
	// map contruct
	int array[8];
	for (int i = 0; i < 8 ; i++) {
		array[i] = i;
	}
	cout << "column-major: " << endl << Map<Matrix<int , 2, 4>>(array) << endl;
	cout << "row-major: " << endl << Map<Matrix<int, 2, 4, RowMajor>>(array) << endl;
	cout << "row-major unaligned using stride: " << endl << Map<Matrix<int ,2 ,4> ,Unaligned ,Stride<1,4>>(array) << endl;

	// using map variables
	typedef Matrix<float, 1, Dynamic> MatrixType;
	typedef Map<MatrixType> MapType;
	typedef Map<const MatrixType> MapTypeConst; // read-only map
	const int n_dims = 5;

	MatrixType m1(n_dims), m2(n_dims);
	m1.setRandom();
	m2.setRandom();
	float *p = &m2(0);
	MapType m2map(p, m2.size());
	MapTypeConst m2mapconst(p, m2.size());
	cout << "matrix m1 : " << endl << m1 << endl;
	cout << "matrix m2 : " << endl << m2 << endl;
	cout << "Squared euclidean distance : " << endl << (m1 - m2).squaredNorm() << endl;
	cout << "Squared euclidean distance ,using map : " << endl << (m1 - m2map).squaredNorm() << endl;
	m2map(3) = 7;
	cout << "matrix m2 new : " << endl << m2 << endl;
	cout << "matrix map2const compile error : " << endl << m2mapconst(2) << endl;

	// change the mapped array
	int data[] = {1,2,3,4,5,6,7,8,9};
	Map<RowVectorXi> v(data ,4);
	cout << "The map vector is " << v << endl;
	new (&v) Map<RowVectorXi>(data + 4, 5);
	cout << "The map new vector is " << v << endl;

	/*
		Reshape and slicing
	*/

	// reshape
	MatrixXf M1(3 ,3);
	M1 <<
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	Map<RowVectorXf> v_8_1(M1.data() ,M1.size());
	cout << "vector v_8_1: " << v_8_1 << endl;
	Matrix<float, Dynamic, Dynamic, RowMajor> M2(M1);
	Map<RowVectorXf> v_8_2(M2.data() ,M2.size());
	cout << "vector v_8_2: " << v_8_2 << endl;

	MatrixXf M_8_4(2,6);
	M_8_4 << 
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12;
	Map<MatrixXf> M_8_5(M_8_4.data() ,6 ,2);
	cout << "M_8_5 :" << endl << M_8_5 << endl;

	// slicing
	RowVectorXf v_8_6 = RowVectorXf::LinSpaced(20 ,0 ,19);
	cout << "input " << v_8_6 << endl;
	Map<RowVectorXf, 0, InnerStride<2>> v_8_7(v_8_6.data() ,v_8_6.size() /2);
	cout << "even " << v_8_7 << endl;

	// strip
	MatrixXf m_8_8 = MatrixXf::Random(3 ,8);
	cout << "column major input : " << endl << m_8_8 << endl;
	Map<MatrixXf, 0, OuterStride<>> m_8_9(m_8_8.data(), m_8_8.rows(), (m_8_8.cols() + 2) / 3,
										OuterStride<>(m_8_8.outerStride() * 3));
	cout << "column major stride 3 : " << endl << m_8_9 << endl;
	typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMajorMatrixXf;
	
	RowMajorMatrixXf m_8_10(m_8_8);
	cout << "row major input : " << endl << m_8_10 << endl;
	Map<RowMajorMatrixXf, 0, Stride<Dynamic, 3>> m_8_11(m_8_10.data(), m_8_10.rows(), (m_8_10.cols() + 2) / 3,
										Stride<Dynamic, 3>(m_8_9.outerStride(), 3));
	cout << "row major stride 3 : " << endl << m_8_11 << endl;

	return 0;
}