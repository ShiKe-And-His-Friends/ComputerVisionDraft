/*
	Space transformations
	//TODO : I need this section but author does not concern. Sad.
*/
#include <iostream>
#include <Eigen/src/Geometry.h>
#include <../src/Geometry.h>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {

	/*
		Geometry

		space transformation

	*/
	
	Transform t;
	t = (AngleAxis(angle, axis));

	Rotation2D<float> rot2(angle_in_radian);

	// The axis vector must to be normalized.
	AngleAxis<float> aa(angle_in_randian ,Vector3f(ax ,ay ,az));

	Quaternion<float> q;
	q = AngleAxis<float>(angle_in_radian ,axis);

	Scaling(sx ,sy);
	Scaling(sx ,sy ,sz);
	Scaling(s);
	Scaling(vecN); //vecN is vector

	Translation<float ,2>(tx ,ty);
	Translation<float ,3>(tx ,ty);
	Translation<float ,N>(s);
	Translation<float ,N>(vecN);

	// N demins affine
	Transform<float ,N ,Affine> t = concatenation_of_any_transformations;
	Transform<float ,3 ,Affine> t = Translation3f(p) * AngleAxisf(a ,axis) * Scaling(s);

	// N demins (pure ,rotations ,scaling ,etc)
	Matrix<float ,N> t = concatenation_of_any_transformations;
	Matrix<float ,2> t = Rotation2Df(a) * Scaling(s);
	Matrix<float ,3> t = AngleAxisf(a ,axis) * Scaling(s);

	// rule: Quaternion first
	Rotation2Df r;
	r = Matrix2f(..); // assumes a pure rotation matrix
	AngleAxisf aa;
	aa = Quaternionf(..);
	AngleAxisf aa;
	aa = Matrix3f(..);
	Matrix2f m ;
	m = Rotation2Df(..);
	Matrix3f m ;
	m = Quaternionf(..);
	Affine3f m;
	m = AngleAxis3f(..);
	Affine3f m;
	m = Translation3f(..);

	// H_{A} = [linear_part ,translation part ; 0.1]
	VectorNf p1,p2;
	p2 = t * p1;

	VecrotNf vec1 ,vec2;
	vec2 = t.linear() * vec1;

	Vectornf n1,n2;
	MattrixNf normalMatrix = t.linear().inverse().transpose(); // A^{-1}^{T}
	n2 = (normalMatrix * n1).normalized();

	// rotate a normalized vector (no scaling ,no shear)
	n2 = t.linear() * n1;
	// OpenGL compatibility 3D
	glLoadMatrixf(t.data());
	// OpenGL compatibility 2D
	Affine3f aux(Affine3f::Identity());
	aux.llinear().topLeftCorner<2 ,2>() = t.lieanr();
	aux.translation().start<2>() = t.translation();
	glLoadMatrixf(aux.data());

	// full read-write access to the internal matrix
	t.matrix() = matN1XN1;
	matN1XN1 = t.matrix();
	t(i ,j) = scalar;
	scalar = t(i ,j);
	t.translation() = vecN;
	vecN = t.translation();
	t.linear() = matNxN;
	matNxN = t.linear();
	extract the rotation matrix
	matNxN = t.rotation();

	t.tranlate(Vector<tx,ty..>);
	t.pretranlate(Vector<tx,ty..>);
	t.rotate(any_rotation);
	t.prerotate(any_rotation);
	t.scale(Vector(sx ,sy));
	t.prescale(Vector(sx ,sy));

	// only 2D
	t.shear(sx ,sy);
	t.preshear(sx ,sy);

	// EulerAngles
	Matrix3f m;
	m = AngleAxisf(angle1 ,Vector::3f :: UnitX())
		* AngleAxisf(angle2 ,Vector::3f :: UnitY())
		* AngleAxisf(angle3 ,Vector::3f :: UnitZ());

	return 0;
}