from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        name='_mask',
        sources=['common/maskApi.c', '_mask.pyx'],
        include_dirs=[np.get_include(), 'common'],
        # extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='_mask',
    # packages=['pycocotools'],
    # package_dir={'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules=cythonize(ext_modules)
)
