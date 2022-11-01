#include "ball_query.hpp"
#include "group_points.hpp"
#include "interpolate.hpp"
#include "sampling.hpp"
#include "ball_query_score.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME , m) {
    m.def("gather_points" ,&gather_points);
    m.def("gather_points_grab" ,&gather_points_grab);
    m.def("furthest_point_sampling" ,&furthest_point_sampling);

    m.def("three_nn" ,&three_nn);
    m.def("three_interpolate" ,&three_interpolate);
    m.def("three_interpolate_grab" ,&three_interpolate_grab);

    m.def("ball_query" ,&ball_query);
    m.def("ball_query_score" ,&ball_query_score);

    m.def("group_points" ,&group_points);
    m.def("group_points_grad" ,&group_points_grad);

}