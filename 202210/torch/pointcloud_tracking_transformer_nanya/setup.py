""""
    构建本地包的动态库

    # python setup.py build_ext --inplace
    #            -----> _ext.cp38-win_amd64.pyd

"""
import os

from setuptools import setup , find_packages
from torch.utils.cpp_extension import BuildExtension ,CppExtension ,CUDAExtension
import glob

# 作者的工具etw_pythorch_utils
try:
    import builtins
except:
    import __builtin__ as builtins
builtins.__POINTNET2_SETUP__ = True

# 导入环境包，可以跳过
import pointnet2

_ext_src_root = os.getcwd() + '/pointnet2/_ext-src'
_ext_source = glob.glob("{}/src/*cxx".format(_ext_src_root)) + glob.glob("{}/src/*.cu".format(_ext_src_root))
_ext_header = glob.glob("{}/include/*hpp".format(_ext_src_root))

requirements = ["etw_pytorch_utils==1.1.1","h5py" ,"pprint" ,"enum34" ,"future"]

setup(
    name= "pointnet2",
    version= "2.1.1",
    author= "Erik Wijmans",
    packages=find_packages(),
    install_requires = requirements,
    ext_modules=[
        CUDAExtension(
        #CppExtension(
            name = "pointnet2._ext",
            sources = _ext_source,
            include_dirs=_ext_header,
            extra_compile_args = {
                "cxx":["-O1" ,"-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc":["-O1" ,"-I{}".format("{}/include".format(_ext_src_root)]
            },
        )
    ],
    cmdclass= {"build_ext" :BuildExtension}
)
