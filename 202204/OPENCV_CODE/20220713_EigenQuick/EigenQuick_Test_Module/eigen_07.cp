#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

	/*
		storage orders
		column-major & row-major
		no same rules, just codes and match. Also other storage librarys.
	*/
	
	// MatrixXf.data() is a pointer to this first value.

	Matrix<int, 3, 4, ColMajor> Acolmajor;
	Acolmajor << 
		8, 2, 2, 9,
		9, 1, 4, 4,
		3, 5, 4, 5;
	cout << " Acolmajor : " << endl << Acolmajor << endl;
	cout << "In memory (column-major) address:" << endl;
	for (int i = 0; i < Acolmajor.size(); i++) {
		cout << (Acolmajor.data() + i) << " ";
	}
	cout << endl;
	cout << "In memory (column-major) value:" << endl;
	for (int i = 0; i < Acolmajor.size(); i++) {
		cout << *(Acolmajor.data() + i) << " ";
	}
	cout << endl << endl;

	Matrix<int, 3, 4, RowMajor> ARowmajor;
	ARowmajor <<
		8, 2, 2, 9,
		9, 1, 4, 4,
		3, 5, 4, 5;
	cout << " ARowmajor : " << endl << ARowmajor << endl;
	cout << "In memory (row-major) address:" << endl;
	for (int i = 0; i < ARowmajor.size(); i++) {
		cout << (ARowmajor.data() + i) << " ";
	}
	cout << endl;
	cout << "In memory (row-major) value:" << endl;
	for (int i = 0; i < ARowmajor.size(); i++) {
		cout << *(ARowmajor.data() + i) << " ";
	}

	/*
		aligement issues
	*/

	// 1. new class maybe need alignement
	//class Foo {
	//	Eigen::Vector2d v;
	//	public : EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	//};

	// 2. STL + Eigen maybe need alignment
	// std::vector<Eigen::Matrix2f> my_vector
	// struct my_class{ ... Eigen::Matrix2f m; ...}
	// solvtions: std::make_shared std::allocate_shared

	// 3. reference maybe need alignment
	// void func(Eigen::Vector4d v);

	// 4. compuler GCC maybe cause no alignment
	// ...

	// Total solvtions
	// This century computer CPU , no-alignment and alignment may cause NULL error. Eigen inner 16Bytes(128bits) still strength SIMD advantage.
	// Usage EIGEN_DONT_VECTORIZE EIGEN_DISABLE_UNALIGENED_ARRAY_ASSERT make 16Bytes align meanwhile ARM compatible.
	// The assert control more feature.

	return 0;
}