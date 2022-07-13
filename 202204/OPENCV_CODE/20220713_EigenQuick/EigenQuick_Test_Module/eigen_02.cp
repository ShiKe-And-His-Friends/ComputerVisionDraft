#include <iostream>
#include <Eigen/Dense>

using namespace std;

int main(int argc, char** argv) {

	// �������
	Eigen::Matrix2d a;
	a << 1, 2,
		3, 4;
	Eigen::MatrixXd b(2, 2);
	b << 2, 3, 5, 6;

	// ����򵥼���
	cout << "a - b :" << endl << a - b << endl;
	cout << "a + b :" << endl << a + b << endl;
	cout << "a^T :" << endl << a.transpose() << endl;
	cout << "a * b :" << endl << a * b << endl;

	// ��������
	Eigen::Vector3d v(1,2,3);
	Eigen::Vector3d w(1,0,0);
	cout << "-v + w -v :" << endl << -v + w -v << endl;
	cout << "v^T : " << endl << v.transpose() << endl;

	//���������Ⱦ���
	Eigen::MatrixXd m = Eigen::MatrixXd::Random(3 ,3);
	Eigen::MatrixXd l = (m + Eigen::MatrixXd::Constant(3, 3, 1.2)) * 50;
	cout << "m : " << endl << m << endl;
	cout << "l : " << endl << l << endl;

	//��������������
	Eigen::VectorXd vv(3);
	w << 7, 8, 9;
	cout << "w : " << endl << vv << endl;
	cout << "m * vv : " << endl << m* vv << endl;
	cout << "vv* m : " << endl << vv.transpose() * m << endl;

	return 0;
}