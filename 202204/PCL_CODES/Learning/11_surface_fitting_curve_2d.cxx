/***

	Surface  ÇúÏßÄâºÏ

**/
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/surface/on_nurbs/fitting_curve_2d.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_pdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_tdm.h>

#include <pcl/surface/on_nurbs/fitting_curve_2d_apdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_atdm.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

namespace Surface_fitting_curve_2d {

	pcl::visualization::PCLVisualizer viewer("Curve Fitting 2D");

	void printHelp(const char *progName) {
	
		std::cout << "\n Usage: " << progName << " [option] "
			<< "\t -h this help" << "\n"
			<< "\t -pd point distance minization" << "\n"
			<< "\t -td tangent distance minimization" << "\n"
			<< "\t -sd squared distance minimization" << "\n"
			<< "\t -apd asymmetirc point-distance-minimization" << "\n"
			<< "\t -asd asymmetric squared-distanct-minimazation" << "\n"
			<< "\t " << "\n"
			<< std::endl;

	}

};

using namespace Surface_fitting_curve_2d;

int main(int argc, char** argv) {

	std::cout << "Surface fitting curve 2d." << std::endl;
	

	
	if (pcl::console::find_argument(argc ,argv ,"-h") >=0) {
		printHelp(argv[0]);
		return 0;
	}
	

	return 0;
}