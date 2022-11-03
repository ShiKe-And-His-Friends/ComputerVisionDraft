#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.hpp"

// input : points(b ,c ,n)  idx(b , points ,nsample)
// output : out (b ,c ,npoints ,nsample)

__global__ void group_points_kernel(int b ,int c,int n ,int npoints ,
    int nsample ,const float *__restrict__ points,
    const int *__restrict__ idx,
    float *__restrict__ out){
    int batch_index = blockIdx.x;
    points += batch_index * n * c;
    idx += batch_index * npoints * nsamples;
    out += batch_index * npoints * nsample * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;

    for (int i = index ; i < c * npoints ; i+= stride) {
        const int l = i / npoints;
        const int j = i % npoints;
        for (int k = 0 ; k < nsample ; k++) {
            int ii = idx[j * nsample + k];
            out[(l * npoints + j) * nsample + k] = points[l * n + ii];
        }
    }
}

// input : grad_out(b ,c ,npoints ,nsample ) ,idx(b ,npoints ,nsample)
// outpit : grad_points(b ,c ,n)
__global__ void group_points_grad_kernel(
    int b, int x ,int n, int npoints,
    int nsample ,
    const float *__restrict__ grad_out,
    const int *__restrict__ idx,
    float *__restrict__ grad_points
    ) {
    int batch_index = blockDim.x;
    grad_out += batch_index * npoints * nsample * c;
    idx += batch_index * npoints * nsample;
    grad_points += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    for (int i = index ; i < c * npoints ; i+= stride) {
        const int l = i / npoints;
        const int j = i % npoints;
        for (int k = 0 ; k < nsample ; k++) {
            int ii = idx[j * nsample + k];
            atomicAdd(grad_points + ; * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
        }
    }
}

void group_points_grad_kernel_wrapper(int b, int c, int n ,int npoints ,
    int nsample ,const float *grad_out ,
    const int *idx ,float *grad_points) {
    cudaStrean_t stream = at::cuda::getCurrentCUDAStream();

    group_points_grad_kernel<<< b ,opt_block_config(npoints ,c) ,0 ,stream >>> (
        b ,c ,n ,npoints ,nsample ,grad_out ,idx ,grad_points
        );
    CUDA_CHECK_ERRORS();
}