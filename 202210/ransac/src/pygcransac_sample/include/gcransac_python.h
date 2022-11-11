#include <vector>
#include <string>

// A method for estimating a homography matrix given 2D-2D correspondences
int findHomography_(
	// The 2D-2D point correspondences.
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers
	std::vector<bool>& inliers,
	// Output: the found 6D pose
	std::vector<double> &homography,
	// The images' sizes
	int h1, int w1, int h2, int w2,
	// The spatial coherence weight used in the local optimization
	double spatial_coherence_weight,
	// The inlier-outlier threshold
	double threshold,
	// The RANSAC confidence. Typical values are 0.95, 0.99.
	double conf,
	// Maximum iteration number. I do not suggest setting it to lower than 1000.
	int max_iters,
	// Minimum iteration number. I do not suggest setting it to lower than 50.
	int min_iters,
	// A flag to decide if SPRT should be used to speed up the model verification.
	// It is not suggested if the inlier ratio is expected to be very low - it will fail in that case.
	// Otherwise, it leads to a significant speed-up.
	bool use_sprt,
	// Expected inlier ratio for SPRT. Default: 0.1
	double min_inlier_ratio_for_sprt,
	// The identifier of the used sampler.
	// Options:
	//	(0) Uniform sampler
	// 	(1) PROSAC sampler. The correspondences should be ordered by quality (e.g., SNN ratio) prior to calling this function.
	//	(2) Progressive NAPSAC sampler. The correspondences should be ordered by quality (e.g., SNN ratio) prior to calling this function.
	//	(3) Importance sampler from NG-RANSAC. The point probabilities should be provided.
	//	(4) Adaptive re-ordering sampler from Deep MAGSAC++. The point probabilities should be provided.
	int sampler_id,
	// The identifier of the used neighborhood structure.
	// 	(0) FLANN-based neighborhood.
	// 	(1) Grid-based neighborhood.
	int neighborhood_id,
	// The size of the neighborhood.
	// If (0) FLANN is used, the size if the Euclidean distance in the correspondence space
	// If (1) Grid is used, the size is the division number, e.g., 2 if we want to divide the image to 2 in along each axes (2*2 = 4 cells in total)
	double neighborhood_size,
	// A flag determining if space partitioning from
	// Barath, Daniel, and Gabor Valasek. "Space-Partitioning RANSAC." arXiv preprint arXiv:2111.12385 (2021).
	// should be used to speed up the model verification.
	bool use_space_partitioning,
	// The variance parameter of the AR-Sampler. It is used only if that particular sampler is selected.
	double sampler_variance,
	// The number of RANSAC iterations done in the local optimization
	int lo_number);
