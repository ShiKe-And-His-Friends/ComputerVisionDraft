""""
    构建本地包的动态库

    # python setup.py build_ext --inplace

"""
from setuptools import setup , find_packages
from torch.utils.cpp_extension import BuildExtension ,CppExtension ,CUDAExtension
import glob

_ext_src_root = '_ext-src'
_ext_source = glob.glob("{}/src/*cxx".format(_ext_src_root)) + glob.glob("{}/src/*.cu".format(_ext_src_root))
_ext_header = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["h5py" ,"pprint" ,"enum34" ,"future"]

setup(
    name= "pointnet2",
    version= "2.1.1",
    author= "Erik Wijmans",
    packages=find_packages(),
    install_requires = requirements,
    ext_modules=[
        CppExtension(
            name = "pointnet2._ext",
            sources = _ext_source,
            extra_compile_args = {
                "cxx":["-02" ,"-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc":["-02" ,"-I{}".format("{}/include").format(_ext_src_root)]
            },
        )
    ],
    cmdclass= {"build_ext" :BuildExtension}
)
