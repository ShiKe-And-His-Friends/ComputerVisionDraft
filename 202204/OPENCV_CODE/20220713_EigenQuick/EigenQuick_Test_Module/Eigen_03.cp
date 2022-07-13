#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc, char** argv) {

	// matrix resize
	Eigen::MatrixXd x = Eigen::MatrixXd::Random(2,5);
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
	Eigen::MatrixXd mat = Eigen::MatrixXd::Random(6,6);
	cout << "mat :" << endl << mat << endl << endl;
	cout << "mat 's sum :" << endl << mat.sum() << endl << endl;
	cout << "mat 's prod :" << endl << mat.prod() << endl << endl;
	cout << "mat 's mean :" << endl << mat.mean() << endl << endl;
	cout << "mat 's minCoeff :" << endl << mat.minCoeff() << endl << endl;
	cout << "mat 's maxCoeff :" << endl << mat.maxCoeff() << endl << endl;
	cout << "mat 's trace :" << endl << mat.trace() << endl << endl;

	// Indexs of values
	std::ptrdiff_t i, j;
	float minCoeffs = mat.minCoeff(&i ,&j);
	cout << "mat 's minCoeff i " << i << " j " << j << endl;

	// ArrayNt ArrayNNt
	Eigen::ArrayXXd a_2_1( 3,3);
	a_2_1 << 
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	cout << "array abs:" << endl << a_2_1.abs() << endl << endl;
	cout << "array abs sqrt:" << endl << a_2_1.abs().sqrt() << endl << endl;

	// Matrix tranform with ArrayNNt
	Eigen::MatrixXf m = Eigen::MatrixXf::Random(3,3);
	Eigen::ArrayXXf a = Eigen::ArrayXXf::Random(3,3);
	cout << "matrix to array:" << endl << m.array() << endl << endl;
	cout << "array to matrix:" << endl << a.matrix() << endl << endl;

	// Block
	cout << "matrix block:" << endl << m.block<2,2>(1 ,1) << endl << endl;
	for (int i = 1; i <= 3; ++i)
	{
		cout << "Block of size " << i << "x" << i << endl;
		cout << m.block(0, 0, i, i) << endl
			<< endl;
	}

	// 

	return 0;
}