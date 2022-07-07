#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>


using namespace std;
using namespace cv;

class ClibaHelp
{

public :

	void calcChessboards(const Size &chessboardSize, vector<Point3f> &corners);

	void decomposeMatrix(const Mat &mat1, const Mat &mat2, Size &patternSize
		,vector<Point3f> &axis, vector<Point2f> &corners1, vector<Point2f> &corners2);

};

