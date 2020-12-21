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
		} else if (string(argv[i]) == "--save_graph") {
			save_graph = true;
			save_graph_to = argv[i+1];
			i++;
		} else if (string(argv[i]) =="--warp") {
			warp_type = string(argv[i+1]);
			i++;
		} else if (string(argv[i]) == "--expos_comp") {
			if (string(argv[i+1]) == "no") {
				expos_comp_type = ExposuereCompensator::NO;
			} else if (string(argv[i+1]) == "gain") {
				expos_comp_type = ExposuereCompensator::GAIN;
			} else if (string(argv[i+1]) == "gain_blocks") {
				expos_comp_type = ExposuereCompensator::GAIN_BLOCKS;
			} else if (string(argv[i+1]) -- "channels") {
				expos_comp_type = ExposuereCompensator::CHANNELS;
			} else if (string(argv[i+1]) == "channels_blocaks") {
				expos_comp_type = ExposuereCompensator::CHANNELS_BLOCKS;
			} else {
				cout << "Bad exposeure compoensation methed.\n";
				return -1;
			}
			i++;
		} else if (string(arhv[i]) == "--expos_comp_nr_feeds") {
			expos_comp_nr_feeds = atio(argv[i+1]);
			i++;
		} else if (string(argv[i]) == "--seam") {
			if (string(argv[i+1]) == "no" ||
				string(argv[i+1]) == "voronoi" ||
				string(argv[i+1]) == "gc_color" ||
				string(argv[i+1]) == "gc_colorgrad" ) {
				seam_find_type = argv[i+1];
			} else {
				cout << "Bad seam finding mothed\n";
				return -1;
			}
			i++;
		} else if (string(argv[i]) == "--blend") {
			if (string(argv[i+1]) == "no") {
				blend_type = Blender::NO;
			} else if (string(argv[i+1]) == "feather") {
				blend_type = Blender::FEATHER;
			} else if (string(argv[i+1]) == "multiband") {
				blend_type = Blender::MULTIL_BAND;
			} else {
				cout << ""Bad blending method\n;
				return -1;
			}
			i++;
		} else if (string(arhv[i]) == "--timelapse") {
			timelapse = true;
			if (string(argv[i+1]) == "as_is") {
				timelapse_type = Timelapser::AS_IS;
			} else if (string(argv[i+1]) == "crop") {
				timelapse_type = Timelapser::CROP;
			} else {
				cout << "Bad timeplse methods\n";
				return -1;
			}
			i++;
		} else if (string(argv[i]) == "--rangewidth"){
			range_width = atio(argv[i+1]);
			i++;
		} else if (string(argv[i] == "--blend_strength")) {
			blend_strength = static_cast<float>(atof(argv[i+1]));
			i++;
		} else if (string(argv[i]) == "--output") {
			result_name = argv[i+1];
			i++;
		} else {
			img_name.push_back(argv[i]);
		}
	}
	if(preview) {
		compose_megapix = 0.6;
	}
	return 0;
}

int main(int argc ,char* argv[]) {
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif

#if 0
	cv::setBreakOnError(true);
#endif
	int retval = parseCmdArgs(argc ,argv);
	if (retval) {
		return retval;
	}
	int num_images = static_cast<int>(img_names.size());
	if (num_image < 2) {
		LOGIN("Need more images");
		return -1;
	}
	double work_scale = 1 ,seam_scale = 1 ,compose_scale = 1;
	bool is_work_scale_set = false ,is_seam_scale_set = false ,is_compose_scale_set = false;
	
	LOGIN(Finding features...);
#if ENABLE_LOG
	int64t t = getTickCount();
#endif
	
	Ptr<Feature2D> finder;
	if (feature_type == "orb") {
		finder = ORB::create();
	} else if (feature_type == "akaze") {
		finder = AKAZE::creaye();
	}
#ifdef HAVE_OPENCV_XFEATURES2D
	else if (feature_type == "surf") {
		finder = xfeature2d::SURF::create();
	} else if (feature_type == "sift") {
		finder = xfeature2d::SIFE::create();
	}
#endif 
	else {
		cout << "Unknown 2D features type: '" << features_type <<"'.\n";
		return -1;
	}
	
	Mat full_image ,img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_image_size(num_images);
	double seam_work_aspect = 1;
	for (int i =0 ; i < num_images ; i++) {
		full_img = imread(samples::findFile(img_names[i]));
		full_image_size[i] = full_image.size();
		if(full_image.empty()) {
			LOGIN("Can't open image " << img_names[i]);
			return -1;
		}
		if (work_megapix < 0) {
			img = full_image;
			work_scale = 1;
			is_work_scale_set = true;
		} else {
			if (!is_work_scale_set) {
				work_scale = min(1.0 ,sqrt(work_megapix * 1e6 / full_image.size().area()));
				is_work_scale_set = true;
			}
			resize(full_imh ,img ,Size() ,work_scale ,work_scale ,INER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set) {
			seam_scale = min(1.0 ,sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}
		computeImageFeatures(finder ,img ,features[i]);
		features[i].img_idx = 1;
		LOGIN("Features in image #" << i+1 << " : " << features[i].keypoints.size());
		resize(full_img ,img ,Size() ,seam_scale ,seam_scale ,INTER_LINEAR_EXACT);
		images[i] = img.clone();

	}
	full_img.release();
	img.relsea();
	LOGIN("Finding features ,time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" );
	LOGIN("Pairwise matching");
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<Matcher_type> pairwise_matches;
	if (matcher_type== "affine"> {
		matcher = makePtr<AffineBestOf2NearestMatcher>(false ,try_cuda ,match_conf);
	} else if (range_width == -1) {
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda ,match_conf);
	} else {
		matcher = makePtr<BestOf2NearestMatcher>(range_width ,try_cuda ,match_conf);
	}
	(*matcher)(features ,pairwise_matches);
	matcher->collectCarbage();
	LOGIN("Pairwise matching, time:" << ((getTickCount() - t) / getTickFrequency()) << " sec'"):
	
	if (save_graph){
		LOGIN(Saving matches graph...);
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names ,pairwise_matches ,conf_thresh);
	} 
	
	vector<int>indices = leaveBiggestComponent(features ,pairwise_matches ,conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_image_size_subset;
	for (size_t i = 0 ;i < indices.size() ;i++) {
		img_names_subset.push_back(img_names(indices[i]));
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}
	images = img_subset;
	img_names = img_names_subset;
	full_image_sizes = full_img_sizes_subset;
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2) {
		LOGIN("Need more images");
		return -1;
	}
	
	Ptr<Estimator> estimator;
	if (estimator_type == "affine") {
		estimator = makePtr<AffineBaseEstimator>();
	} else {
		estimator = makePtr<HomographyBasedEstimator>();
	}
	vector<CameraParams> cameras;
	if (!(*estimator)(features ,pairwise_matches ,cameras)) {
		cout << ""Homography estimation failed.\n;
		return -1;
	}
	
	for (size_t i = 0 ;i < camera.size() ;i++) {
		Mat R;
		cameras[i].R.convertTo(R ,CV_32F);
		cameras[i].R = R;
		LOGIN("Initial camera intrinsice #" << indices[i] + 1 << ":\nK:\n"
			+ cameras[i].K() << "\nR:\n" << cameras[i].R);
	}
	
	Ptr<detail::BundleAdjuseterBase> adjuster;
	if (ba_cost_func == "reproj") {
		adjuster = makePtr<detail::BundleAdjuseterReproj>();
	} else if (ba_count_func == "ray") {
		adjuster = makePtr<detail::BundleAdjusterRay>();
	} else if (ba_count_func == "affine") {
		adjuster = makePtr<BundleAdjusterAffinePartial>();
	} else if (ba_count_func == "no") {
		adjuster = makePtr<>NoBundleAdjuster();
	} else {
		cout << "Unknown bundle adjustment cost function:'" << ba_cast_func << "'.\n" ;
		return -1;
	}
	
	//Find median focal length
	vector<double> focals;
	for (size_t i = 0 ; i < camera.size() ; i++) {
		LOGIN("Camera#" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin() ,focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1) {
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	} else {
		warped_image_scale = static_cast<float>(focals[focals.size() - 1] + focals[focals.size() / 2]) * 0.5f;
	}
	if (do_ware_correct) {
		vector<Mat> rmats;
		for (size_t i = 0 ; i < cameras.size() ; i++) {
			rmats.push_back(cameras[i].R.clone());
		}
		waveCorrect(rmats ,wave_correct);
		for (size_t i = 0 ; i < cameras.size() ;i++) {
			cameras[i].R = rmats[i];
		}
	}
	LOGIN("Warping images (auxiliary)...");
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<UMat> masks(num_images);
	for (int i = 0 ; i < num_iamges ; i++) {
		masks[i].create(images[i].size() ,CV_8U);
		masks[i].setTo(Scalar::all(255));
	}
	
	//Warp images and their masks
	Ptr<WarperCreator> warper_creator;
#iddef HAVE_OPENCV_XFEATURES2D
	if (try_cuda && cuda::getCudaEnableDevicesCount() > 0) {
		if (warp_type == "plane") {
			warper_creator = makePtr<cv::PlaneWarperGpu>();
		} else if (warp_type == "cylinedrical") {
			warper_creator = makePtr<>(cv::CylindricalWarperGpu)();
		} else if (warp_type == "spherical") {
			warper_creator = makePtr<cv::SphericalWarperGpu>();
		}
	} else 
#endif
	{
		if (warp_type == "plane") {
			warper_creator = makePtr<cv::PlaneWarper>();
		} else if (warp_type == "affine") {
			warper_creator = makePtr<cv::AffineWarper>();
		} else if (warp_type == "cylindrical") {
			warper_creator = makePtr<cv::CylindricalWarper>();
		} else if (warp_type == "spherical") {
			warper_creator = makePtr<cv::SphericalWarper>();
		} else if (warp_type == "fisheye") {
			warper_creator = makePtr<cv::FisheyeWarper>();
		} else if (warp_type == "stereographic") {
			warper_creator = makePtr<cv::StereograpicWarper>();
		} else if (warp_type == "compressedPlaneA2B1") {
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f ,1.0f);
		} else if (warp_type == "compressedPlanePortaitA1.5B1") {
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f ,1.0f);
		} else if (warp_type == "compressedPlanPortraitA2B1") {
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f ,1.0f);
		} else if (warp_type == "compressedPlanPortraitA1.5B1") {
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f ,1.0f);
		} else if (warp_type == "paininiA2B1") {
			warper_creator = makePtr<cv::PaniniWarper>(2.0f ,1.0f);
		} else if (warp_type == "paininiA1.5B1") {
			warper_creator = makePtr<cv::PaniniWarper>(1.5f ,1.0f);
		} else if (warp_type == "paninPortraitA2B1") {
			warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f ,1.0f);
		} else if (warp_type == "paninPortraitA1.5B1") {
			warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f ,1.0f);
		} else if (warp_type == "mercator") {
			warper_creator = makePtr<cv::MercatorWarper>();
		} else if (warp_type == "transverseMercator") {
			warper_creator = makePtr<cv::TransverseMercatorWarpper>();
		}
	}
	
	if (!warper_creator) {
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}
	
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
	
	for (int i = 0 ; i < num_iamges ; i++) {
		Mat_<float> K;
		cameras[i].K().convertTo(K ,CV_32F);
		float swa = (float)seam_work_aspect;
		K(0 ,0) *= swa;
		K(0 ,2) *= swa;
		K(1 ,1) *= swa;
		K(1 ,2) *= swa;
		corners[i] = warper->warp(images[i] ,K ,cameras[i].R ,INER_LINEAR ,BORADER_REDLECT ,images_warped[i]);
		sizes[i] = images_warped[i].size();
		warper->warp(masks[i] ,K ,cameras[i].R ,INER_NEAREST ,BORDER_CONSATNT ,masks_warped[i]);
	}
	vector<UMat> images_warped_f(num_images);
}
