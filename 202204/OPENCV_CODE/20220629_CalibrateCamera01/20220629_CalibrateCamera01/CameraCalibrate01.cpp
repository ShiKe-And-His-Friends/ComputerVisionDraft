#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;

const char* help = 
	"{ help h | | need help }"
	"{ image | | image check }";

int main(int argc, char* argv[]) {

	cv::CommandLineParser parser(argc, argv, help);
	
	if (parser.has("image")) {
		cout << "has image" << endl;
		return -1;
	}
	else {
		cout << "no image" << endl;
		return -2;
	}

	string input = parser.get<string>("@image");
	cout << "Input image uri : " << input << endl;

	if (input == "") {
		cout << "Input image empty. " << input << endl;
		return -1;
	}

	cv::Mat imageMat = cv::imread(input, cv::IMREAD_GRAYSCALE);

	return 0;
}
