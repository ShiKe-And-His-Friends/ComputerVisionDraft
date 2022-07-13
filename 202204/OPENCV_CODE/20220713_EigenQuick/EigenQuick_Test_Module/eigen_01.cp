#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc ,char** argv) {

	// �������
	Eigen::Matrix2d a;
	a << 1, 2,
		3,4;
	Eigen::MatrixXd b(2, 2);
	b << 2, 3, 5, 6;

	// ����򵥼���
	cout << "a - b " << endl << a - b << endl;
	cout << "a + b " << endl << a + b << endl;
	cout << "a^T " << endl << a.transpose()<< endl;
	cout << "a * b " << endl << a * b << endl;

	// ��������
	Eigen::Vector3d v();

	return 0;
}