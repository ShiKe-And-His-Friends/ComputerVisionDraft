#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc ,char** argv) {

	// 输入矩阵
	Eigen::Matrix2d a;
	a << 1, 2,
		3,4;
	Eigen::MatrixXd b(2, 2);
	b << 2, 3, 5, 6;

	// 矩阵简单计算
	cout << "a - b " << endl << a - b << endl;
	cout << "a + b " << endl << a + b << endl;
	cout << "a^T " << endl << a.transpose()<< endl;
	cout << "a * b " << endl << a * b << endl;

	// 输入向量
	Eigen::Vector3d v();

	return 0;
}