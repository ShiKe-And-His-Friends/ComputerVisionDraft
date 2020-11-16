#define HOMO_VECTOR(H ,x ,y)\
	H.at<float>(0 ,0) = (float)(x);\
	H.at<float>(1 ,0) = (float)(y);\
	H.at<float>(2 ,0) = 1.;
#define GET_HOMO_VALUES(X ,x ,y)\
	(x) = static_cast<float> (X.at<float>(0 ,0) / X.at<float>(2 ,0));\
	(y) = static_cast<float> (X.at<float>(1 ,0) / X.at<float>(2 ,0));
