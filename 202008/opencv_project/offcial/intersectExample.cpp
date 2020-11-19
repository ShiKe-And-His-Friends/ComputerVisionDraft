/**
 * Authout: Steve Nicholson
 * A program that illukstrates intersectConvexConvex in various scenarios
 * */

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

// craete a vector of points describing a rectangle with the given corners
static vector<Point> makeRectangle(Point topLeft ,Point bottomRight) {
	vector<Point> rectangle;
	rectangle.push_back(topLeft);
	rectangle.push_back(Point(bottomRight.x ,topLeft.y));
	rectangle.push_back(bottomRight);
	rectangle.push_back(Point(topLeft.x ,bottomRight.y));
	return rectangle;
}

static vector<Point> makeTriangle(Point point1 ,Point point2 ,Point point3) {
	vector<Point> triangle;
	triangle.push_back(point1);
	triangle.push_back(point2);
	triangle.push_back(point3);
	return triangle;
}

