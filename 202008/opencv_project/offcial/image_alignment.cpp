/*
 * This sample demonstrates the use of the function findTransfromECC that implements the image alignment ECC algorithem
 * 
 * The demo loads an image (default to fruits.jpg) and it artificaially creates a template image based on the given motion type.
 * When to image are given ,the first image is the input image and the second on defines the template image.
 * In the latter case ,you can also parse the warp's initialization.
 * Input and output warp files consist of raw warp (transform) elements
 * Authors: G.Evangelidis ,INRIA ,Grenoble ,France
 *          M.Asbach ,Fraunhofer IAIS ,St. Augusion ,Germany
 */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <stdio.h>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

static void help(void);
static int readWrap(string iFilename ,mat& warp ,int motionType);
static int saveWarp(string fileName ,const Mat& warp ,int motionType);
static void draw_warped_roi(Mat& image ,const int width ,const int height ,Mat& W);

#define HOMO_VECTOR(H ,x ,y)\
	H.at<float>(0 ,0) = (float)(x);\
	H.at<float>(1 ,0) = (float)(y);\
	H.at<float>(2 ,0) = 1.;
#define GET_HOMO_VALUES(X ,x ,y)\
	(x) = static_cast<float> (X.at<float>(0 ,0) / X.at<float>(2 ,0));\
	(y) = static_cast<float> (X.at<float>(1 ,0) / X.at<float>(2 ,0));
	
const std::string keys = 
	"{@inputImage		| fruits.jpg	| input image filename}"
	"{@templateImage	| 				| template image filename (optionnal)}"
	"{@inputWarp		|				| input warp (maxtrix) filename (optional)}"
	"{n numofIte		| 50 			| ECC's iterations}"
	"{}"
