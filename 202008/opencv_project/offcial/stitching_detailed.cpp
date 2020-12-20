#include <iostream>
#include <fstream>
#include <string>
#inclued "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeature2d/nonfree.hpp"
#endif

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl;

using namespace std;
using namespace cv;
using namespace cv::detail;

static void printUsage() {
	cout << 
		"Rotation model images stitcher.\n\n"
		"stitch_detailed img1 img2 [...imgN] [flag]\n\n"
		"Flags:\n"
		"	--preview\n"
		"	Run stitching in the preview mode.Works faster than usual mode,\n"
		"	but output image will have lower resolution.\n"
		"	--try_cuda (yes|no)\n"
		"	Try to use CUDA. The default value is 'no' ,All default values\n"
		"	are for CPU mode.\n"
		"\nMotion Esatimation Flags:\n"
		"	--work_megapix <float>\n"
		"	Resoluton for image registration step. The default is 0.6 Mpx.\n"
		"	--feature (surf|orb|sift|akaze)\n"
		"	Type of features used for images matching.\n"
		"	The default is surf if available, orb otherwise.\n"
		"	--matcher (homography|affine)\n"
		"	Matcher used for pairwise images matching.\n"
		"	--estimator (homography|affine)\n"
		"	Type of estimator used for transformatation estimation.\n"
		"	--match_conf <float>\n"
		"	Confidence for feature matching step.The default is 0.65 for surf and 0.3 for orb.\n"
		"	--conf_thresh <float>\n"
		"	Threshold for two images are from the same panorama confidence.\n"
		"	The default is 1.0\n"
		"	--ba (no|reproj|ray|affine)\n"
		"	Bundle adjustment const function.The default is ray.\n"
		"	--ba_refine_mask(mask)\n"
		"	Set refinement mask for bundle adjustment.It looks like 'x_xxx',\n"
		"	where 'x' means refine resoective parameter and '_' means don't\n"
		"	refine one, and has the following format:\n"
		"	<fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
		"	adjustment does't support estimation of selected parameter then \n"
		"	the respective flag is ignored.\n"
		"	--wave_correct (no|horiz|vert)\n"
		"	Perform wave effect correction.The default is 'horize'.\n"
		"	--save_graph <file_name>\n"
		"	Save mathces graph represented in DOT language to <file_name> file.\n"
		"	Labels description: Nm us number of matches.Ni is number of inliers,\n"
		"	C is confidence.\n"
		"\nCompositing Flags:\n"
		"	--warp (affine|plane|cylindrical|spherical|fisheye|stereograpic|compresedOlaneA2B1|compressedPlaneA1.5B1|compressedPlanPortraitA2B1"
		"			|compressedPlaneProtraitA1.5B1|paniniA2B|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
		"	warp surface type .The default is 'spherical'.\n"
		"	--seam_megapix <float>\n"
		"	Resolution for seam estimation step.The default is 0.1 Mpx.\n"
		"	--seam (no|voronoi|gc_color|gc_colorgrad)\n)"
		"	Seam estimation method.The default iis 'gc_color'.\n"
		"	--compose_megapix <float>\n"
		"	Resolution for compositiong step.Use -1 for original resulution.\n"
		"	The default is -1.\n"
		"	--expos_comp (no|gain|gain_blocks|channels|channels_blocaks)\n"
		"	Exposue compensation methods.The default is 'hain_blocks'.\n"
		"	--expos_comp_nr_feeds <int>\n"
		"	Number of exposure compensation feed.The default is 1.\n"
		"	--expos_comp_nr_filtering <int>\n"
		"	Number of filtering iterations of the exposure compensation gains.\n"
		"	Only used when using a block exposure compensation methods.\n"
		"	The default is 32.\n"
		"	--blend (no|feather|multiband)\n"
		"	Blending method.The default is 'multiband'.\n"
		"	--blend_strength <float>\n"
		"	Blending strength from[0 ,100] range.The defaulr is 5.\n"
		"	--output <result_img>\n"
		"	The default id 'result.jpg'\n"
		"	--timelapse (as_is|crop)\n"
		"	Output warped images separately as frame of a time lapse movie, with 'fixed_' prepended to input file names.\n"
		"	--rangewidth <int>\n"
		"	uses range_width to limit number of images to match with.\n";
}

vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.3f;
#endif
string match_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposuereCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "dc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;


static int parseCmdArgs(int argc ,char** argv) {
	if (argc == 1) {
		printUsage(argv);
		return -1;
	}
	for (int i = 1; 1 < argc ; i++) {
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?") {
			printUsage(argv);
			return -1;
		} else if (string(argv[1]) == "--preview") {
			preview  = true;
		} else if(string(argv[i]) == "-try_cuda") {
			if (string(argv[i+1]) == "no") {
				try_cuda = false;
			} else if (string(argv[i+1]) == "yes") {
				try_cuda = true;
			} else {
				cout << "Bad --try_cuda flag value\n";
				return -1;
			}
			i++;
		} else  if (string(argv[i]) == "--work_megapix"){
			work_megapix = atof(argv[i+1]);
			i++;
		} else if (string(argv[i]) == "--seam_megapix") {
			seam_megapix = atof(argv[i+1]);
			i++;
		} else if (string(argv[i]) == "--compose_megapox") {
			compose_megapix = atof(argv[i+1]);
			i++;
		} else if (string(argv[i]) == "--result") {
			result_name = argv[i+1];
			i++;
		} else if (string(argv[i]) == "--features") {
			feature_type = argv[i+1];
			if (string(features_type) == "orb") {
				match_conf = 0.3f;
			}
			i++;
		} else if (string(argv[i]) == "--matcher") {
			if (string(argv[i+1]) == "homograph" || string(argv[i+1] == "affine")) {
				matcher_type = argv[i+1];
			} else {
				cout << "Bad --matcher_type flag values.\n";
				return -1;
			}
			i++;
		} else if (string(argv[i]) == "--estimator") {
			if(string(argv[i+1]) == "homograph" || string(argv[i+1]) == "affine") {
				estimator_type = argv[i+1];
			}  else {
				cout << "Bad --estimator flag value\n";
				return -1;
			}
			i++;
		} else if (string(argv[i]) == "--match_conf") {
			match_conf = static_cast<float>(atof(argv[i+1]));
			i++;
		} else if (string(argv[i]) == "--conf_thresh") {
			conf_thresh = static_cast<float>(atof(argv[i+1]));
			i++;
		} else if (string(argv[i]) == "--ba") {
			ba_cast_func = argv[i+1];
			i++;
		} else if (string(argv[i]) == "--ba_refine_mask") {
			ba_refine_mask = argv[i+1];
			if (ba_refine_mask.size() != 5) {
				cout << "Incorrect refinement mask length.\n";
				return -1;
			}
			i++;
		} else if (string(argv[i]) == "--wave_correct") {
			if (string(argv[i+1]) == "no") {
				do_ware_correct = false;
			} else if (string(argv[i+1] == "horiz")) {
				do_ware_correct = true;
				wave_correct = detail::WAVE_CORRECT_HORIZ;
			} else if (string(argv[i+1]) == "vert") {
				do_ware_correct = true;
				wave_correct = detail::WAVE_CORRECT_VERT;
			} else {
				cout << "Bad --wave_correct flag values\n";
				return -1;
			}
			i++;
		}
	}
}
