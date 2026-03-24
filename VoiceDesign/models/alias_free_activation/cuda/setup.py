from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="anti_alias_activation_cuda",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "anti_alias_activation_cuda",
            ["anti_alias_activation.cpp", "anti_alias_activation_cuda.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)   
