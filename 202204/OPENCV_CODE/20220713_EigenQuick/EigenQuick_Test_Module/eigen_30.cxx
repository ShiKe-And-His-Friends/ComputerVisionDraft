/*
	Sparse linear Algebra
*/
#include <iostream>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat; // column-major sparse matrix
typedef Eigen::Triplet<double> T;

void insertCoefficient(int id, int i, int j, double w, std::vector<T> &coeffs,
	Eigen::VectorXd &b, const Eigen::VectorXd &boundary) {
	int n = int(boundary.size());
	int id1 = i + j * n;
	if (i == -1 || i == n) {
		b(id) -= w * boundary(j);
	}
	else if (j == -1 || j == n) {
		b(id) -= w * boundary(i);
	}
	else {
		coeffs.push_back(T(id ,id1 ,w));
	}
}

void buildProblem(std::vector<T> &coefficients , Eigen::VectorXd &b ,int n) {
	b.setZero();
	Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n ,0 ,3.14).sin().pow(2);
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < n; ++i)
		{
			int id = i + j * n;
			insertCoefficient(id, i - 1, j, -1, coefficients, b, boundary);
			insertCoefficient(id, i + 1, j, -1, coefficients, b, boundary);
			insertCoefficient(id, i, j - 1, -1, coefficients, b, boundary);
			insertCoefficient(id, i, j + 1, -1, coefficients, b, boundary);
			insertCoefficient(id, i, j, 4, coefficients, b, boundary);
		}
	}
}

int main(int argc, char** argv) {

	/*
		Sparse matrix manipulations

		1.values 2.InnerIndices 3.OuterStarts 4.InnerNNZs

	*/
	// no need
	
	return 0;
}