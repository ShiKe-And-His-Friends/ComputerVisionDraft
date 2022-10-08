/***

	visualization

	## important

**/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

namespace visualization_demo {

	void printHelp(const char* console_dir) {
		std::cout <<
			"\nUsage : " << console_dir << " <option> [sample.pcd]" << std::endl 
			<< "-h help \n"
			<< "-s Simple visualisation example \n"
			<< "-r RGB colour visualisation example\n"
			<< "-c Custom colour visualizaiton example\n"
			<< "-n Normals visualization example\n"
			<< "-a Shapes visualization example\n"
			<< "-v Viewroot example\n"
			<< "-i Interaction customization example \n"
			<< std::endl;
	}


	boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleViewers(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "sample cloud");
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		return (viewer);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbViewers(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("rgb viewer"));
		viewer->setBackgroundColor(0,0,0);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud ,rgb ,"sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE ,7 ,"sample cloud");
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		return (viewer);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> customColourViewers(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("custom colour viewer"));
		viewer->setBackgroundColor(0 ,0 ,0);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud ,0,255,0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud ,single_color ,"sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "sample cloud");
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		return (viewer);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsViewers(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("normal viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "sample cloud");
		viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud ,normals ,10 ,0.05 ,"normals");
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		return (viewer);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> shapesViewers(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("sahpes viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "sample cloud");
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();

		viewer->addLine<pcl::PointXYZRGB>(cloud->points[0] ,cloud->points[cloud->size() -1] ,"line");
		viewer->addSphere(cloud->points[0] ,0.2 ,0.5 ,0.5 ,0.0 ,"sphere");

		// ax+by+cz+d=0
		pcl::ModelCoefficients coeffs;
		coeffs.values.push_back(0.0);
		coeffs.values.push_back(0.0);
		coeffs.values.push_back(1.0);
		coeffs.values.push_back(0.0);
		viewer->addPlane(coeffs ,"plane");

		coeffs.values.clear();
		coeffs.values.push_back(0.3);
		coeffs.values.push_back(0.3);
		coeffs.values.push_back(0.0);
		coeffs.values.push_back(0.0);
		coeffs.values.push_back(1.0);
		coeffs.values.push_back(0.0);
		coeffs.values.push_back(5.0);
		viewer->addCone(coeffs ,"cone");

		return (viewer);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsViewers(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud
			,pcl::PointCloud<pcl::Normal>::ConstPtr normals1, pcl::PointCloud<pcl::Normal>::ConstPtr normals2) {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewport viewer"));
		viewer->initCameraParameters();
		int v1(0);
		viewer->createViewPort(0.0 ,0.0 ,0.5 ,1.0 ,v1);
		viewer->setBackgroundColor(0 ,0 ,0 ,v1);
		viewer->addText("Radius:0.01" ,10,10 ,"v1 text",v1);
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud ,rgb ,"sample cloud1" ,v1);
		int v2(1);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
		viewer->addText("Radius:0.1", 10, 10, "v2 text", v2);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud ,0,255,0);
		viewer->addPointCloud<pcl::PointXYZRGB>(cloud, single_color, "sample cloud2", v2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");
		viewer->addCoordinateSystem(1.0);
		// add normals to multi groups
		viewer->addPointCloudNormals<pcl::PointXYZRGB ,pcl::Normal>(cloud ,normals1 ,10 ,0.05 ,"normals1" ,v1);
		viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals1, 10, 0.5, "normals2", v2);

		return (viewer);
	}

	unsigned int text_id = 0;

	void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event ,void *viewer_void) {
		pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);
		if (event.getKeyCode() == 'r' && event.keyDown()) {
			std::cout << "r was pressed => removing all text" << std::endl;
			char str[512];
			for (unsigned int i = 0; i < text_id; i++) {
				sprintf(str ,"text#%03d" ,i);
				viewer->removeShape(str);
			}
			text_id = 0;
		}
	}

	void mouseEventOccurred(const pcl::visualization::MouseEvent & event ,void *viewer_void) {
		pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);
		if (event.getButton() == pcl::visualization::MouseEvent::LeftButton
			&& event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease) {
		
			std::cout << "left mouse button released at position " << event.getX() << " " << event.getY() << std::endl;
			char str[512];
			sprintf(str ,"text#%03d" ,text_id++);
			viewer->addText("clicked here" ,event.getX() ,event.getY() ,str);
		}
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationViewers() {
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("interaction viewer"));
		viewer->setBackgroundColor(0 ,0,0);
		viewer->registerKeyboardCallback(keyboardEventOccurred ,(void *)viewer.get());
		viewer->registerMouseCallback(mouseEventOccurred, (void*)viewer.get());
		return (viewer);
	}
};

using namespace visualization_demo;

int visualization_demo_sphare(int argc ,char **argv) {

	std::cout << "visualization demo" << std::endl;
	if (pcl::console::find_argument(argc, argv, "-h") >=0) {
		printHelp(argv[0]);
		return -1;
	}

	bool simple(false), rgb(false), custom_c(false), normals(false), shapes(false),
		viewports(false), interaction_customization(false);
	if (pcl::console::find_argument(argc,argv,"-s") >=0) {
		simple = true;
		std::cout << "Simple visualization example." << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-c") >= 0) {
		custom_c = true;
		std::cout << "Custom colour visualization example." << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-r") >= 0) {
		rgb = true;
		std::cout << "RGB colour visualization example." << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-n") >= 0) {
		normals = true;
		std::cout << "Normals visualization example." << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-a") >= 0) {
		shapes = true;
		std::cout << "Shapes visualization example." << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-v") >= 0) {
		viewports = true;
		std::cout << "Viewports visualization example." << std::endl;
	}
	else if (pcl::console::find_argument(argc, argv, "-i") >= 0) {
		interaction_customization = true;
		std::cout << "Interaction customization visualization example." << std::endl;
	}
	else {
		printHelp(argv[0]);
		return -2;
	}

	// create sample point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::cout << "Genarating example point clouds" << std::endl;
	uint8_t r(255), g(15), b(15);

	for (float z(-1.0f); z <= 1.0; z += 0.05f) {
		for (float angle(0.0); angle <= 360.0;angle+=5.0) {
			pcl::PointXYZ basic_point;
			basic_point.x = 0.5 * cosf(pcl::deg2rad(angle));
			basic_point.y = sinf(pcl::deg2rad(angle));
			basic_point.z = z;
			basic_cloud_ptr->points.push_back(basic_point);

			pcl::PointXYZRGB point;
			point.x = basic_point.x;
			point.y = basic_point.y;
			point.z = basic_point.z;

			uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
				static_cast<uint32_t>(g) << 8 |
				static_cast<uint32_t>(b)
			);
			point.rgb = *reinterpret_cast<float*>(&rgb);
			point_cloud_ptr->points.push_back(point);
		}
		if (z < 0.0) {
			r -= 12;
			g += 12;
		}
		else {
			g -= 12;
			b += 12;
		}
	}
	basic_cloud_ptr->width = (int)basic_cloud_ptr->points.size();
	basic_cloud_ptr->height = 1;
	point_cloud_ptr->width = (int)point_cloud_ptr->points.size();
	point_cloud_ptr->height = 1;

	// calculate surface normals with a search radius of 0.05
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
	ne.setInputCloud(point_cloud_ptr);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	ne.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.05);
	ne.compute(*cloud_normals1);
	// calculate surface normals with a search radius of 0.1
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.1);
	ne.compute(*cloud_normals2);

	// visualization
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	if (simple) {
		viewer = simpleViewers(basic_cloud_ptr);
	} else if (rgb) {
		viewer = rgbViewers(point_cloud_ptr);
	} else if (custom_c) {
		viewer = customColourViewers(basic_cloud_ptr);
	} else if (normals) {
		viewer = normalsViewers(point_cloud_ptr , cloud_normals2);
	} else if (shapes) {
		viewer = shapesViewers(point_cloud_ptr);
	} else if (viewports) {
		viewer = viewportsViewers(point_cloud_ptr ,cloud_normals1 ,cloud_normals2);
	} else if (interaction_customization) {
		viewer = interactionCustomizationViewers();
	}

	while (!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

	return 0;
}