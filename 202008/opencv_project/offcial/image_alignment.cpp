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
	"{e epsilon			| 0.0001		| ECC's convergence epsilon}"
	"{o outputWarp		| outWarp.ecc 	| ouput warp (maxtrix) filename}"
	"{m motionType		| affine		| type of motion (translation, euclidean ,affine ,hemography)}"
	"{v verbose			| 1				| display initial and final images}"
	"{w warpedImfile	| warpedECC.png	| warped input image}"
	"{h help			| print help message}";
	
static void help (void) {
	cout << "\nThis file demonstrates the use of the ECC image alignment algorthm. Whrn one image"
		" us given ,the template image is artifically formed by a random warp. When both images"
		" are given ,the initialization of the warp by command line parsing is possible."
		" If inputWarp is missing, the identity transformation initializes the algorithm. \n" << endl;
		
	cout << "\nUsage example (one image):\n /image_alignment youtInput.png youtTemplate.png"
		" -m=euclidean -e=1e-6 -N=70 -v=1 \n" <<endl;
		
	cout <<"\nUsage example (two images with initialization): \n /image_alignment youtInput.png yourTemplate.png"
		" yorInitialWarp.ecc -o=outWarp.ecc -m=homography -e=1e-6 -M=70 -v=1 -w=yourFinalImage.png \n" <<endl;
}

static int readWarp(string iFilename ,Mat& warp ,int motionType) {
	// it reads for file a specific number of raw values;
	// 9 values for homography ,6 otherwise
	CV_Assert(warp.type() == CV_32FC1);
	int numOfElements;
	if (motionType == MOTION_HOMOGRAPHY) {
		numOfElements = 9;
	} else {
		numOfElements = 6;
	}
	
	int i;
	int ret_values;
	ifstream myfile(iFilename.c_str());
	if (myfile.is_open()) {
		float* matPtr = warp.ptr<float>(0);
		for ( i = 0 ; i < numOfElements ; i++) {
			myfile >> matPtr[i];
		}
		ret_values = 1;
	} else {
		cout << "Unable to open file " << iFilename.c_str() << endl;
		ret_values = 0;
	}
	return ret_value;
}

static int saveWarp(string fileName ,const Mat& warp , int motionType) {
	// it saves the raw matrix elements in a file
	CV_Assert(warp.type() == CV_32FC1);
	const float* matPtr = warp.ptr<float>(0);
	int ret_value;
	ofstream outfile(fileName.c_str());
	if (!outfile) {
		cerr << "error in saving "
			<< "Couldn't open file '" << fileName.c_str() << "'!" << endl;''
		ret_values = 0;
	} else {
		// save the warp's elements
		outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << endl;
		outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << endl;
		if (motionType == MOTION_HOMOGRAPHY) {
			outfile << matPtr[6] << " " << matPtr[7] << " " << matPtr[8] << endl;
		}
		ret_values = 1;
	}
	return ret_values;
}

static void draw_warped_roi(Mat& image ,const int width ,const int height ,Mat& W) {
	Point2f top_left ,top_right ,bottom_left ,bottom_right;
	Mat H = Mat(3 ,1 ,CV_32F);
	Mat U = Mat(3 ,1 ,CV_32F);
	Mat warp_mat = Mat::eye(3 ,3 ,CV_32F);

	for (int y = 0 ; y < W.rows ; y++ ) {
		for (int x = 0 ; x < W.cols ; x++ ) {
			warp_mat.at<float>(y ,x) = W.at<float>(y ,x);
		}
	}
	// warp the corners pf rectangle

	// top-left
	HOMO_VECTOR(H ,1 ,1);
	gemm(warp_mat ,H ,1 ,0 ,0 ,U);
	GET_HOMO_VALUES(U ,top_left.x ,top_left.y);

	// top-right
	HOMO_VECTOR(H ,width ,1);
	gemm(warp_mat ,H ,1 ,0 ,0 ,U);
	GET_HOMO_VALUES(U ,top_right.x ,top_right.y);

	// bottom-left
	HOMO_VECTOR(H ,1 ,height);
	gemm(warp_mat ,H ,1 ,0 ,0 ,U);
	GET_HOMO_VALUES(U ,bottom-left.x ,bottom-left.y);

	//bottom-right
	HOMO_VECTOR(H ,width ,height);
	gemm(warp_mat ,H ,1 ,0 ,0 ,U);
	GET_HOMO_VALUES(U ,bottom_right.x ,bottom_right.y);

	//draw the warped perimeter
	line(image ,top_left ,top_right ,Scalar(255));
	line(image ,top_right ,bottom_right ,Scalar(255));
	line(image ,bottom_right ,bottom_left ,Scalar(255));
	line(image ,bottom_left ,top_left ,Scalar(255));
}

int main(const int argc ,char* argv[]) {
	CommandLineParser parser(argc ,argv ,keys);
	parser.about("ECC demo");

	parser.printMessage();
	help();

	string imgFile = parser.get<string>(0);
	string tempImaFile = parser.get<string>(1);
	string inWarpFile = parser.get<string>(2);

	int number_of_iterations = parser.get<int>("n");
	double termination_eps = parser.get<double>("e");
	string warpType = parser.get<string>("m");
	int verbose = parser.get<int>("v");
	string finalWarp = parser.get<string>("o");
	string warpImFile = parser.get<string>("w");

	if (!parser.check()) {
		parser.printErrors();
		return -1;
	}
	if (!(warpType == "translation" || warpType == "euclidean" 
			|| warpType == "affine" || warpType == "homography" )) {
		cerr << "Invalid motion transformation " << endl;
		return -1 ;
	}

	int mode_temp;
	if (warpType == "translation") {
		mode_temp = MOTION_TRANSLATION;
	} else if (warpType == "euclidean") {
		mode_temp = MOTION_EDCLIDEAN;
	} else if (warpType == "affine") {
		mode_temp = MOTION_AFFINE;
	} else {
		mode_temp = MOTION_HOMOGRAPHY;
	}

	Mat inputImage = imread(samples::findFile(imageFile) ,INREAD_GRAYSCALE);
	if (inputImage.empty()) {
		cerr << "Unable to load the input Image" << endl;
		return -1;
	}

	Mat targer_image;
	Mat template_image;

	if (tempImaFile != "") {
		inputImage.copyTo(targer_image);
		template_image = imread(samples::findFile(tempImaFile) ,IMREAD_GRAYSCALE);
		if (template_image.empty()) {
			cerr << "Unable to load the template image" << endl;
			return -1;
		}
	} else {
		// apply random warp to inpit image
		
	}

}
