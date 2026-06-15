from setuptools import setup, Extension

my_ext = Extension(
    name="vfhe._my_ext",
    sources=["src/vfhe/c_src/my_ext.c"],
    include_dirs=["src/vfhe/include"],
)

setup(
    ext_modules=[my_ext]
)