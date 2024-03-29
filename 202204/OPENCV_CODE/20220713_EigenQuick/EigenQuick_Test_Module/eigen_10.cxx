/*
	Dense linear problem and  decomposition
*/
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

	/*
		linear algebra and decomposition
	*/
	
	// basic linear solving
	Matrix3f A_1_1;
	Vector3f b_1_1;
	A_1_1 << 1, 2, 3, 4, 5, 6, 7, 8, 10;
	b_1_1 << 3, 3, 4;
	
	Vector3f x_1_1 = A_1_1.colPivHouseholderQr().solve(b_1_1);
	cout << "solution1 : " << endl << x_1_1 << endl << endl;
	ColPivHouseholderQR<Matrix3f> dec_1_1(A_1_1);
	Vector3f x_1_2 = dec_1_1.solve(b_1_1);
	cout << "solution2 : " << endl << x_1_1 << endl << endl;

	// Decomposition			Method					Require on the Matrix	speed	accuracy
	// PartialPivLU				partialPivLU()			Invertiable				++		++
	// FullPivLU				fullPivLU()				None					-		--
	// HouseholderQR			householderQR()			None					++		++
	// ColPivHouseholderQR		colPivHouseholderQR()	None					+		-
	// FullPiveHouseholderQR	fullPivHouseholderQR()	None					-		+++
	// CompleteOrthogonal		completeOrthogonal-		None					-		+++	
	//	-Decomposition			Decompostition()
	// LLT						llt()					Positive definite		+++		+
	// LDLT						ldlt()					Positive or negative	++		++
	//													- semidefinite			
	// BDCSVD					bdcSvd()				None					-		+++

	Matrix2f A_1_2, b_1_2;
	A_1_2 << 2, -1, -1, 3;
	b_1_2 << 1, 2, 3, 1;
	Matrix2f x_1_3 = A_1_2.ldlt().solve(b_1_2);
	cout << "solution : " << endl << x_1_3 << endl << endl;

	// check if a solution really exists
	MatrixXd A_1_4 = MatrixXd::Random(30 ,30);
	MatrixXd B_1_4 = MatrixXd::Random(30, 15);
	MatrixXd X_1_4 = A_1_4.fullPivLu().solve(B_1_4);
	double relative_error = (A_1_4 * X_1_4 - B_1_4).norm() / B_1_4.norm();
	cout << "relative error : " << endl << relative_error << endl;

	// computing eigen values and eigen vectors
	Matrix2f A_1_5;
	A_1_5 << 1, 2, 2, 3;
	SelfAdjointEigenSolver<Matrix2f> eigensolver(A_1_5);
	if (eigensolver.info() != Success) {
		abort();
	}
	cout << " eigen values of a are: " << eigensolver.eigenvalues() << endl;
	cout << " eigen vector of a are: " << eigensolver.eigenvectors() << endl;

	// computing inverse and determinant (small matrix only)
	Matrix3f A_1_6;
	A_1_6 << 
		1, 2, 1,
		2, 1, 0,
		-1, 1, 2;
	cout << "matrix " << A_1_6 << endl;
	cout << "determinant " << A_1_6.determinant() << endl;
	cout << "inverse " << A_1_6.inverse() << endl;

	// least squares solving
	MatrixXf A_1_7 = MatrixXf::Random(3 ,2);
	cout << "matrix " << endl << A_1_7 << endl;
	VectorXf b_1_7 = VectorXf::Random(3);
	cout << "vector " << b_1_7 << endl;
	cout << "least-squares solutions is : \n"
		<< A_1_7.bdcSvd(ComputeThinU | ComputeThinV).solve(b_1_7) << endl;

	// Separating the computation
	Matrix2f A_1_8, b_1_8;
	LLT<Matrix2f> llt;
	A_1_8 << 2, -1, -1, 3;
	b_1_8 << 1, 2, 3, 1;
	cout << "matrix " << endl << A_1_8 << endl;
	cout << "right hand side " << endl << b_1_8 << endl;
	llt.compute(A_1_8);
	cout << "solvtion " << endl << llt.solve(b_1_8) << endl;

	A_1_8(1, 1)++;
	llt.compute(A_1_8);
	cout << "new solvtion " << endl << llt.solve(b_1_8) << endl;
	
	HouseholderQR<MatrixXf> qr(50 ,50);
	MatrixXf A = MatrixXf::Random(50 ,50);
	qr.compute(A);

	// rank revealing decomposition
	Matrix3f A_1_9;
	A_1_9 <<
		1, 2, 5,
		2, 1, 4,
		3, 0, 3;
	FullPivLU<Matrix3f> lu_decomp(A_1_9);
	cout << "the rank of matrix is " << lu_decomp.rank() << endl;
	cout << "a matrix whose columns from a basis of the null-space of matrix :" << endl
		<< lu_decomp.kernel() << endl;
	cout << "a matrix whose columns from a basis of the columns-space of matrix :" << endl
		<< lu_decomp.image(A_1_9) << endl;
	lu_decomp.setThreshold(1e-5);
	cout << "a matrix whose columns from a basis of the columns-space of matrix new:" << endl
		<< lu_decomp.image(A_1_9) << endl;

	/*
		text catelog of decompositions
	*/
	// *

	// 1. LDLT algorithm has two varitances.
	// 2. a eigenvalue like SVD algorithm based on iteration speeds.
	// 3. openMP used servals fo core speed acceleration.

	/*
		solving linear least squares systems
		Ax = b , Overdetermined equation
	*/
	
	// using the svd decomposition
	MatrixXf A_2_10 = MatrixXf::Random(3,2);
	cout << "matrix" << endl << A_2_10 << endl;
	VectorXf b_2_10 = VectorXf::Random(3);
	cout << "right hand vector " << b_2_10 << endl;
	MatrixXf x_2_10 = A_2_10.bdcSvd(ComputeThinU | ComputeThinV).solve(b_2_10);
	cout << "the least-squares solution is " << endl << x_2_10 << endl;

	// using the QR decompostion
	MatrixXf A_2_11 = MatrixXf::Random(3, 2);
	cout << "matrix" << endl << A_2_11 << endl;
	VectorXf b_2_11 = VectorXf::Random(3);
	cout << "right hand vector " << b_2_11 << endl;
	MatrixXf x_2_11 = A_2_11.colPivHouseholderQr().solve(b_2_11);
	cout << "the solution using the QR decomposition:" << endl << x_2_11 << endl;

	// using normal equation
	MatrixXf A_2_12 = MatrixXf::Random(3 ,2);
	VectorXf b_2_12 = VectorXf::Random(3);
	VectorXf x_2_12 = (A_2_12.transpose() * A_2_12).ldlt().solve(A_2_12.transpose() * b_2_12);
	cout << "the solution " << endl << x_2_12 << endl;

	/*
		inplace decomposition
		complete pivoting(full pivoting) / partial pivoting
	*/
	// inplace matrix decomposition
	MatrixXd A_2_13(2 ,2);
	A_2_13 << 2, -1, 1, 3;
	cout << "Here is " << endl << A_2_13 << endl;
	PartialPivLU<Ref<MatrixXd>> lu(A_2_13); // linking reference
	cout << "here is input matrix after decomposition " << endl << A_2_13 << endl;
	cout << "Here is the matirx storing the L and U factor:\n" << lu.matrixLU() << endl;

	MatrixXd A_2_14(2 ,2);
	A_2_14 << 2, -1, 1, 3;
	VectorXd b_2_14(2);
	b_2_14 << 1, 2;
	lu.compute(A_2_14);
	VectorXd x_2_14 = lu.solve(b_2_14);
	cout << "Residual : " << (A_2_14 * x_2_14 - b_2_14).norm() << endl;

	A_2_13 << 3, 4, -2, 1;
	x_2_14 = lu.solve(b_2_14);
	A_2_14 = A_2_13;
	lu.compute(A_2_13);
	x_2_14 = lu.solve(b_2_14);
	cout << "Residual : " << (A_2_14 * x_2_14 - b_2_14).norm() << endl;

	MatrixXd A_2_15(2,2);
	A_2_15 << 5, -2, 3, 4;
	lu.compute(A_2_15);
	cout << "here is the input matrix after decompostion:" << endl << A_2_15 << endl;
	x_2_14 = lu.solve(b_2_14);
	cout << "Residual : " << (A_2_15 * x_2_14 - b_2_14).norm() << endl;

	/*
		Bench mark of dense deompostion
		text of different result in hardware
	*/
	// *

	return 0;
}