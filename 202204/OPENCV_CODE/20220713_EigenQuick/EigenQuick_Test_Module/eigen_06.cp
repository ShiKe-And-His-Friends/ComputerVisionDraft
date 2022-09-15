#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

	/*
		aliasing
		A = AB  /  a = a^T b  /  A = A*A
		eval() : using aliasing maybe
		noalias() : force no using aliasing
	*/
	
	// aliasing
	MatrixXi m_9_1(3 ,3);
	m_9_1 <<
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	m_9_1.bottomRightCorner(2, 2) = m_9_1.topLeftCorner(2 ,2); // 5
	cout << "after alisaing matrix is " << endl << m_9_1 << endl;
	//Matrix2i a_9_1;
	//a_9_1 <<1, 2, 3, 4;
	//cout << "a_9_1 : " << endl << a_9_1 << endl;
	//a_9_1 = a_9_1.transpose();

	// EIGEN_NO_DEBUG : control assert

	// resolving aliasing issus
	MatrixXi m_9_3(3, 3);
	m_9_3 <<
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	m_9_3.bottomRightCorner(2, 2) = m_9_3.topLeftCorner(2, 2).eval(); // 5
	cout << "after alisaing matrix is " << endl << m_9_3 << endl;
	MatrixXf m_9_2(2,3);
	m_9_2 << 1, 2, 3, 4 ,5,6;
	cout << "initial matrix " << endl << m_9_2 << endl;
	m_9_2.transposeInPlace();
	cout << "matrix after transposeinplace" << endl << m_9_2 << endl;
	/*
		MatrixBase::adjointInPlace()
		DenseBase::reverseInPlace()
		LDLT::solveInPlace()
		TriangularView::solveInPlace()
		DenseBase::transposeInPlace()
	*/

	// aliasing and component wise operation
	MatrixXf m_9_4(4, 4), m_9_5(4, 4);
	m_9_4.setRandom();
	m_9_5 = m_9_4;
	cout << "m_9_4 : " << endl << m_9_4 << endl;
	m_9_4 = 2 * m_9_4;
	cout << "m_9_4 x 2 : " << endl << m_9_4 << endl;
	m_9_4 = m_9_4 - MatrixXf::Identity(4,4);
	cout << "m_9_4 - E : " << endl << m_9_4 << endl;
	ArrayXXf a_9_4 = m_9_4;
	a_9_4 = a_9_4.square();
	cout << "a_9_4 after square : " << endl << a_9_4 << endl << endl;

	cout << "m_9_5 : " << endl << m_9_5 << endl;
	m_9_5 = (2 * m_9_5 - MatrixXf::Identity(4, 4)).array().square();
	cout << "m_9_5 after squre: " << endl << m_9_5 << endl;

	// aliasing and matrix multiplicate
	MatrixXf m_9_7(2, 2);
	m_9_7 << 2, 0, 0, 2;
	m_9_7 = m_9_7 * m_9_7;
	cout << " m_9_7 : " << endl << m_9_7 << endl;

	MatrixXf m_9_8(2, 2) , m_9_9(2, 2);
	m_9_8 << 2, 0, 0, 2;
	m_9_9.noalias() = m_9_8 * m_9_8;
	cout << " m_9_9 : " << endl << m_9_9 << endl;

	MatrixXf m_9_10(2, 2);
	m_9_10 << 2, 0, 0, 2;
	m_9_10.noalias() = m_9_10 * m_9_10; //my computer also no error
	cout << " m_9_10 : " << endl << m_9_10 << endl;

	MatrixXf m_9_11(2, 2), m_9_12(3, 2);
	m_9_12 << 2, 0, 0, 3, 1, 1;
	m_9_11 << 2, 0, 0, -2;
	m_9_11 = (m_9_12 * m_9_11).cwiseAbs();
	cout << " m_9_11 : " << endl << m_9_11 << endl;

	MatrixXf m_9_13(2, 2), m_9_14(3, 2);
	m_9_14 << 2, 0, 0, 3, 1, 1;
	m_9_13 << 2, 0, 0, -2;
	m_9_13 = (m_9_14 * m_9_13).eval().cwiseAbs();
	cout << " m_9_13 : " << endl << m_9_13 << endl;

	return 0;
}