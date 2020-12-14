/**
 * pca.cpp
 * Author: kevin Hughes
 * This proggram demonstrates how to use OpenCV PCA with abort
 * specified amount of variance to retain. The effect
 * is illustrated further by using trackbar to 
 * change the value for retained varaince.
 * The first 15 faces of the AT&T face data set:
 * http://www.cl.cam.uk/research/dtg/attarchive/facedatabase.html
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

static void read_imgList(const string& filename ,vector<Mat>& images) {
	std::ifstream file(filename.c_str() ,ifstream::in);
	if (!file) {
		string error_message = "No file.";
		CV_Error(Error::StsBadArg ,error_message);
	}
	string line;
	while(getline(file ,line)) {
		images.push_back(imread(line ,0));
	}
}

static Mat toGrayscale(InputArray _src) {
	Mat src = _src.getMat();
	if (src.channels() != 1) {
		CV_Error(Error::StsBadArg ,"Only Matrices with one channel are supported");
	}
	Mat dst;
	cv::normalize(_src ,dst ,0 ,255 ,NORM_MINMAX ,CV_8UC1);
	return dst;
}

struct params {
	Mat data;
	int ch;
	int rows;
	PCA pca;
	string winName;
}

static void onTrackbar(int pos ,void* ptr) {
	cout << "Retained Variance = " << pos << "%  ";
	cout << "re-calculating PCA ..." << std::flush;
	double var = pos / 100.0;
	struct param *p = (struct param *)ptr;
	p->pca = PCA(p->data ,cv::Mat() ,PCA::DARA_AS_ROW ,var);
	Mat point = p->pca.project(p->data.row(0));
	Mat reconstruction = p->pca.backProject(point);
	reconstruction = reconstruction.reshape(p->ch ,p->rows);
	reconstruction = toGrayscale(reconstruction);
	imshow(p->winName ,reconstruction);
	cout << "done!  # of principal components: " << p->pca.eigenvectoes.rows << endl;
}

int main(int argc ,char** argv) {
	cv::CommandLineParser parser(argc ,argv ,"{@input || image list}{help h||show help message}");
if (parser.has("help")) {
	parser.printMessage();
	exit(0);
}

string imgList = parser.get<string>("@input");
if (imgList.empty()) {
	parser.printMessage();
	exit(1);
}

vector<Mat> images;
try {
	read_imgList(imgList ,images);
} catch (const cv::EXception& e) {
	cerr << "Error opening file \" " << imgList << "\".Reason: " << e.msg << endl;
	exit(1);
}

if (images.size() <= 1) {
	string error_message = "This demo needs to least 2 images to works,";
	CV_Error(Error::StsError ,error_message);
	
}

Mat data = formatImageForPCA(images);
PCA pca(data ,cv::Mat() ,PCA::DATA_AS_ROW ,0.95);
Mat point = pca.project(data,row(0));
Mat reconstruction = pca.backProject(point);
reconstruction = reconstruction.reshape(images[].channels() ,images[0].rows);
reconstruction toGrayscale(reconstruction);
string winName = ""Reconstruction | press 'q' to quit";
namedWindow(winName ,WINDOW_NORMAL);
params p;
p.data = data;
p.ch = images[].channels();
p.rows = images[0].rows;
p.pca = pca;
p.winName = winName;
int pos = 95;
createTrackbar("Retained Variance (%)" ,winName ,&pos ,100 ,onTrackbar ,(void*)&p);
imshow(winName ,reconstruction);
char key = 0;
while(key != 'q') {
	key = (char)waitKey();
}
return 0;
}