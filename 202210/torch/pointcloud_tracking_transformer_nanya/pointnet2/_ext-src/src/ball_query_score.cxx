#include "ball_score_search.hpp"
#include "utils.hpp"

void query_ball_point_score_kernel_wrapper(
    int b ,int n ,int m ,float radius ,int nsample ,const float *new_xyz ,const float *xyz ,const float *score,
    float * unique_score);


/***
    #TODO debug
**/
void query_ball_point_score_kernel_wrapper_CPU(
    int b ,int n ,int m ,float radius ,int nsample ,const float *new_xyz ,const float *xyz ,const float *score,
    float * unique_score){
    int batch_index = blockIdx.x;
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    score += batch_index * n;
    unique_score += m * batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    float radius2 = radius * radius;
    for(int  j = index ; j < m ; j += stride) {
        float new_x = new_xyz[j * 3 + 0];
        float new_y = new_xyz[j * 3 + 0];
        float new_z = new_xyz[j * 3 + 0];
        for (int k = 0 ,cnt = 0 ; k < n && cnt < nsample ; k++) {
            float x = xyz[k * 3 + 0];
            float y = xyz[k * 3 + 1];
            float z = xyz[k * 3 + 2];
            float s = score[k];
            float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z -z);
            if (d2 < radius2) {
                unique_score[j] += s;
                cnt++;
            }
        }
    }
}

at::Tensor ball_query_score(at::Tensor new_xyz ,at::Tensor xyz ,at::Tensor score ,
    const float* radius ,const int nsample){
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(score);
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_FLOAT(score);

    if(new_xyz.type().is_cuda() {
        CHECK_CUDA(xyz);
        CHECK_CUDA(score);
    }

    at::Tensor unique_score =
        torch::zeros(new_xyz.size(0) ,new_xyz.size(1) ,at::device(new_xyz.device()).dtype(at::ScalarType::Float) );
    if (new_xyz.type().is_cuda() ){
        query_ball_point_score_kernel_wrapper(
            xyz.size(0) ,xyz.size(1) ,new_xyz.size(1),
            radius, nsample ,new_xyz.data<float>(),
            xyz.data<float>() ,score.data<float>() ,unique_score.data<float>()
            );
    } else {
        //AT_CHECK(false ,"Query Ball : CPU not support");
        query_ball_point_score_kernel_wrapper(
            xyz.size(0) ,xyz.size(1) ,new_xyz.size(1),
            radius, nsample ,new_xyz.data<float>(),
            xyz.data<float>() ,score.data<float>() ,unique_score.data<float>()
            );

    }
}