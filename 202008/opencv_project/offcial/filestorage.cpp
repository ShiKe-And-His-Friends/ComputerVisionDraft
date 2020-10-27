#include "opencv2/core.hpp"
#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::endl;
using std::cerr;
using std::ostream;
using namespace cv;

static void help (char** av) {
	cout << "\nfilestorage_sample demonstrate the usage of the usage of the opencv serialization functionality.\n"
		<< "usage:\n"
		<< av[0] << " outputfile.yml.gz\n"
		<< "\nThis program demonstrates the use of FileStorage for serialization ,that is in use <<and>> in OpenCv\n"
		<< "For example ,how to create a class and have it serialize ,but also how to read and write matrices.\n"
		<< "FileStorage allows you to serialize to various formats specified by the file and type."
		<< "\nYou should try using different file extensions.(e.g yaml yml xml xml.gz yaml.gz etc...)\n" << endl;
}

struct MyDate {
	myDate():
		A(1) ,X(0) ,id()
	{ }

	explicit MyDate(int):
		A(97) ,X(CV_PI) ,id("mydata1234")
	{ }

	int A;
	double X;
	string id;
	void write (FileStorage& fs) const //Write serialization for this clas
	{
		fs << "{" << "A" << A << " X" << X << " id " << id << "}";
	}

	void read (const FileNode& node)  //Read serilization for this class 
	{
		A = (int)node["A"];
		X = (double)node["X"];
		id = (string)node["id"];
	}
};



