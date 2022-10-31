#include <torch/extension.h>

#define CHECK_IS_FLOAT(x)\
do\
{
    AT_CHECK(x.scalar_type() == at::ScalarType::Float, \
        # x " must be a float tensor")\
} while (0)