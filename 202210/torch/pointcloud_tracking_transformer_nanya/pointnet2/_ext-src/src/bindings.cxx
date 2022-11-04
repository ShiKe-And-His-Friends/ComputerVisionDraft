#include "ball_query.hpp"
#include "group_points.hpp"
#include "interpolate.hpp"
#include "sampling.hpp"
#include "ball_query_score.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME , m) {
    m.def("gather_points" ,&gather_points ,"gather points");
    m.def("gather_points_grad" ,&gather_points_grad ,"gather points grab");
    m.def("furthest_point_sampling" ,&furthest_point_sampling ,"furtheast point sampling");

    m.def("three_nn" ,&three_nn ,"three nn");
    m.def("three_interpolate" ,&three_interpolate ,"three interpolate");
    m.def("three_interpolate_grad" ,&three_interpolate_grad ,"three interplate grab");

    m.def("ball_query" ,&ball_query ,"ball query");
    m.def("ball_query_score" ,&ball_query_score ,"ball query score");

    m.def("group_points" ,&group_points ,"group socre");
    m.def("group_points_grad" ,&group_points_grad ,"group points grad");

}