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
#include <pcl/surface/on_nurbs/fitting_curve_2d_sdm.h>

#include <pcl/surface/on_nurbs/fitting_curve_2d_apdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_asdm.h>
#include <pcl/surface/on_nurbs/fitting_curve_2d_atdm.h>

#include <pcl/surface/on_nurbs/triangulation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

namespace Surface_fitting_curve_2d {

	pcl::visualization::PCLVisualizer viewer("Curve Fitting 2D");

	void PointCloud2Vector2d(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud ,pcl::on_nurbs::vector_vec2d &data) {
		for (unsigned i = 0; i < cloud->size(); i++) {
			pcl::PointXYZ& p = cloud->at(i);
			if (!std::isnan(p.x) && !std::isnan(p.y)) {
				data.push_back(Eigen::Vector2d(p.x ,p.y));
			}
		}
		std::cout << "Point vecotr 2d : " << data.size() << std::endl;
	}

	void VisualizeCurve(ON_NurbsCurve& curve, double r, double g, double b, bool show_cps) {
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::on_nurbs::Triangulation::convertCurve2PointCloud(curve ,cloud ,8);
		for (size_t i = 0; i < cloud->size() -1; i++) {
			pcl::PointXYZRGB& p1 = cloud->at(i);
			pcl::PointXYZRGB& p2 = cloud->at(i+1);
			std::ostringstream os;
			os << "line_" << r << "_" << g << "_" << b << "_" << i;
			viewer.addLine<pcl::PointXYZRGB>(p1 ,p2 ,r ,g ,b ,os.str());
		}

		if (show_cps) {
			pcl::PointCloud<pcl::PointXYZ>::Ptr cps(new pcl::PointCloud<pcl::PointXYZ>);
			for (int i = 0; i < curve.CVCount(); i++) {
				ON_3dPoint cp;
				curve.GetCV(i ,cp);

				pcl::PointXYZ p;
				p.x = float(cp.x);
				p.y = float(cp.y);
				p.z = float(cp.z);
				cps->push_back(p);
			}
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(cps ,255 * r ,255 * g ,255 * b);
			viewer.addPointCloud<pcl::PointXYZ>(cps ,handler ,"cloud_cpss");
		}

	}

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

int curve_2d_fitting(int argc, char** argv) {

	std::cout << "Surface fitting curve 2d." << std::endl;
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PCDReader reader;
	reader.read(argv[1] ,*cloud);
	
	if (pcl::console::find_argument(argc ,argv ,"-h") >=0) {
		printHelp(argv[0]);
		return 0;
	}
	bool pd(false), td(false), sd(false), apd(false), atd(false), asd(false);
	if (pcl::console::find_argument(argc,argv ,"-pd") >= 0) {
		pd = true;
		std::cout << "pdm\n" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-sd") >= 0) {
		sd = true;
		std::cout << "sdm\n" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-apd") >= 0) {
		apd = true;
		std::cout << "apdm\n" << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-td") >= 0) {
		atd = true;
		std::cout << "stdm\n" << std::endl;
	}
	else if (pcl::console::find_argument(argc,argv ,"-asd") >= 0) {
		asd = true;
		std::cout << "asdm\n" << std::endl;
	}
	else {
		printHelp(argv[0]);
		return 0;
	}

	// spatial tranform : [y-o-z]->[x-o-z]
	pcl::PointCloud<pcl::PointXYZ>::iterator it_1;
	for (it_1 = cloud->begin(); it_1 != cloud->end();) {
		float x = it_1->x;
		float y = it_1->y;
		float z = it_1->z;
		it_1->x = z;
		it_1->y = y;
		it_1->z = x;
		it_1++;
	}

	// data type convert
	pcl::on_nurbs::NurbsDataCurve2d data;
	PointCloud2Vector2d(cloud ,data.interior);

	// initialize curve
	unsigned order(3); // k-level
	unsigned n_control_points(10);
	ON_NurbsCurve curve = pcl::on_nurbs::FittingCurve2dSDM::initCPsNurbsCurve2D(order ,data.interior);

	// curve fitting
	if (pd) {
		pcl::on_nurbs::FittingCurve2dPDM::Parameter curve_params;
		curve_params.smoothness = 0.000001;
		curve_params.rScale = 1.0;
		pcl::on_nurbs::FittingCurve2dPDM fit(&data ,curve);
		fit.assemble(curve_params);
		fit.solve();
		VisualizeCurve(fit.m_nurbs ,1.0 ,0.0 ,0.0 ,false);
	}
	else if (td) {
		pcl::on_nurbs::FittingCurve2dTDM::Parameter curve_params;
		curve_params.smoothness = 0.000001;
		curve_params.rScale = 1.0;
		
		pcl::on_nurbs::FittingCurve2dTDM fit(&data, curve);
		fit.assemble(curve_params);
		fit.solve();
		VisualizeCurve(fit.m_nurbs ,1.0 ,0 ,0 ,false);
	}
	else if (sd) {
		pcl::on_nurbs::FittingCurve2dSDM::Parameter curve_params;
		curve_params.smoothness = 0.000001;
		curve_params.rScale = 1.0;

		pcl::on_nurbs::FittingCurve2dSDM fit(&data ,curve);
		fit.assemble(curve_params);
		fit.solve();
		VisualizeCurve(fit.m_nurbs ,1.0 ,0 ,0 ,false);
	}
	else if (apd) {
		pcl::on_nurbs::FittingCurve2dAPDM::Parameter curve_paramters;
		curve_paramters.smoothness = 0.000001;
		curve_paramters.rScale = 1.0;

		pcl::on_nurbs::FittingCurve2dAPDM fit(&data ,curve);
		fit.assemble(curve_paramters);
		fit.solve();
		VisualizeCurve(fit.m_nurbs ,1.0 ,0 ,0 ,false);
	}
	else if (asd) {
		pcl::on_nurbs::FittingCurve2dASDM::Parameter curve_params;
		curve_params.smoothness = 0.000001;
		curve_params.rScale = 1.0;
		
		pcl::on_nurbs::FittingCurve2dASDM fit(&data,curve);
		fit.assemble(curve_params);
		fit.solve();
		VisualizeCurve(fit.m_nurbs ,1.0 ,0 ,0 ,false);
	}

	viewer.setSize(800 ,600);
	viewer.setBackgroundColor(255 ,255 ,255);
	viewer.addPointCloud<pcl::PointXYZ>(cloud ,"cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR ,0 ,0 ,0 ,"cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,8 ,"cloud");
	viewer.spin();

	return 0;
}