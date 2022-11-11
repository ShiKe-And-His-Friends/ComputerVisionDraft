#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <stdexcept> //运行时异常库
#include "gcransac_python.h"

namespace py = pybind11;

py::tuple findHomography(py::array_t<double>  correspondences_,
                         int h1, int w1, int h2, int w2,
    					 py::array_t<double>  probabilities_,
                         double threshold,
                         double conf,
							double spatial_coherence_weight,
							int max_iters,
							int min_iters,
							bool use_sprt,
							double min_inlier_ratio_for_sprt,
							int sampler,
							int neighborhood,
							double neighborhood_size,
							bool use_space_partitioning,
							int lo_number,
							double sampler_variance,
							int solver)
{
    py::buffer_info buf1 = correspondences_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (solver == 0 && DIM != 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,4], n>=4" );
    }
    if (solver > 0 && DIM != 8) {
        throw std::invalid_argument( "SIFT or affine correspondences should be an array with dims [n,8], n>=4" );
    }
    if (NUM_TENTS < 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,d], n>=4");
    }

    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);
    }

    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);

    int num_inl = 0;

	if (solver == 0) // Point correspondence-based fundamental matrix estimation
		num_inl = findHomography_(
			correspondences,
			probabilities,
			inliers,
			H,
			h1, w1, h2, w2,
			spatial_coherence_weight,
			threshold,
			conf,
			max_iters,
			min_iters,
			use_sprt,
			min_inlier_ratio_for_sprt,
			sampler,
			neighborhood,
			neighborhood_size,
			use_space_partitioning,
			sampler_variance,
			lo_number);
	else if (solver == 1) // SIFT correspondence-based fundamental matrix estimation
		num_inl = findHomographySIFT_(
			correspondences,
			probabilities,
			inliers,
			H,
			h1, w1, h2, w2,
			spatial_coherence_weight,
			threshold,
			conf,
			max_iters,
			min_iters,
			use_sprt,
			min_inlier_ratio_for_sprt,
			sampler,
			neighborhood,
			neighborhood_size,
			sampler_variance,
			lo_number);
	else if (solver == 2) // Affine correspondence-based fundamental matrix estimation
		num_inl = findHomographyAC_(
			correspondences,
			probabilities,
			inliers,
			H,
			h1, w1, h2, w2,
			spatial_coherence_weight,
			threshold,
			conf,
			max_iters,
			min_iters,
			use_sprt,
			min_inlier_ratio_for_sprt,
			sampler,
			neighborhood,
			neighborhood_size,
			sampler_variance,
			lo_number);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> H_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = H_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];

    return py::make_tuple(H_,inliers_);
}


py::tuple findHomography(py::array_t<double>  correspondences_,
                         int h1, int w1, int h2, int w2,
    					 py::array_t<double>  probabilities_,
                         double threshold,
                         double conf,
							double spatial_coherence_weight,
							int max_iters,
							int min_iters,
							bool use_sprt,
							double min_inlier_ratio_for_sprt,
							int sampler,
							int neighborhood,
							double neighborhood_size,
							bool use_space_partitioning,
							int lo_number,
							double sampler_variance,
							int solver)
{
    py::buffer_info buf1 = correspondences_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (solver == 0 && DIM != 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,4], n>=4" );
    }
    if (solver > 0 && DIM != 8) {
        throw std::invalid_argument( "SIFT or affine correspondences should be an array with dims [n,8], n>=4" );
    }
    if (NUM_TENTS < 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,d], n>=4");
    }

    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);
    }

    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);

    int num_inl = 0;
    num_inl = findHomography_(
        correspondences,
        probabilities,
        inliers,
        H,
        h1, w1, h2, w2,
        spatial_coherence_weight,
        threshold,
        conf,
        max_iters,
        min_iters,
        use_sprt,
        min_inlier_ratio_for_sprt,
        sampler,
        neighborhood,
        neighborhood_size,
        use_space_partitioning,
        sampler_variance,
        lo_number);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> H_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = H_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];

    return py::make_tuple(H_,inliers_);
}

PYBIND11_PLUGIN(pygcransac) {
    py::module m("pygcransac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pygcransac
        .. autosummary::
           :toctree: _generate

           //findFundamentalMatrix,
			//findLine2D,
			findHomography,
		   //find6DPose,
		   //findEssentialMatrix,
		   //findRigidTransform,

    )doc");

    m.def("findHomography", &findHomography, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("h1"),
        py::arg("w1"),
        py::arg("h2"),
        py::arg("w2"),
        py::arg("probabilities"),
        py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
        py::arg("spatial_coherence_weight") = 0.975,
        py::arg("max_iters") = 10000,
		py::arg("min_iters") = 50,
	  	py::arg("use_sprt") = true,
	  	py::arg("min_inlier_ratio_for_sprt") = 0.00001,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 0,
		py::arg("neighborhood_size") = 4.0,
		py::arg("use_space_partitioning") = true,
		py::arg("lo_number") = 50,
		py::arg("sampler_variance") = 0.1,
		py::arg("solver") = 0);

    return m.ptr();
}