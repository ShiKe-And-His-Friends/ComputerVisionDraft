#include "ball_query.hpp"
#include "utils.hpp"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius, int nsample, const float *new_xyz, const float *xyz, int *idx);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
	const int nsample) {
	CHECK_CONTIGUOUS(new_xyz);
	CHECK_CONTIGUOUS(xyz);
	CHECK_IS_FLOAT(new_xyz);
	CHECK_IS_FLOAT(xyz);

	if (new_xyz.type().is_cuda()) {
		CHECK_CUDA(xyz);
	}
	at::Tensor idx = torch::zeros({ new_xyz.size(0), new_xyz.size(1), nsample },
		at::device(new_xyz.device()).dtype(at::ScalarType::Int));
	if (new_xyz.type().is_cuda()) {
		query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
			radius, nsample, new_xyz.data<float>(), xyz.data<float>(), idx.data<int>());
	}
	else {
		TORCH_CHECK(false, "CPU not supported");
	}

	return idx;
}