/***
    Python调用C++接口，使用CUDA的GPU加速
    来处理KITTI数据集

    2022-11-01 sk9510
**/

#pragma once
#include <torch/extension.h>
//#include <Aten/cuda/CUDAContext.h>

#define CHECK_CUDA(x) \
  do { \
    AT_CHECK(x.type().is_cuda() ,#x " must be a CUDA tensor"); \
  } while(0)

#define CHECK_CONTIGUOUS(x) \
  do{ \
    AT_CHECK(x.is_contiguous() ,#x" must be a contiguous tensor"); \
  } while(0)

#define CHECK_IS_INT(x) \
  do { \
    AT_CHECK(x.scalar_type() == at::ScalarType::Int, #x" must be an int tensor"); \
  } while(0)

#define CHECK_IS_FLOAT(x)\
do\
{
    AT_CHECK(x.scalar_type() == at::ScalarType::Float, \
        # x " must be a float tensor");\
} while (0)