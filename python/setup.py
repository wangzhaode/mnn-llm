from setuptools import setup, Extension, find_packages
import platform

IS_DARWIN = platform.system() == 'Darwin'
IS_WINDOWS = (platform.system() == 'Windows')

def make_relative_rpath(path):
    """ make rpath """
    if IS_DARWIN:
        return [f'-Wl,-rpath,@loader_path/../../../{path},-rpath,@loader_path/{path}']
    elif IS_WINDOWS:
        return []
    else:
        return ['-Wl,-rpath,$ORIGIN/' + path]

lib_suffix = 'so'
if IS_DARWIN:
    lib_suffix = 'dylib'

packages = find_packages()
lib_files = [('lib',
            [f'../libs/libMNN.{lib_suffix}',
             f'../libs/libMNN_Express.{lib_suffix}',
             f'../build/libllm.{lib_suffix}'])]

module = Extension('cmnnllm',
                  sources=['./mnnllm.cpp'],
                  include_dirs=['../include'],
                  library_dirs=['../build', '../libs'],
                  extra_compile_args=['-std=c++17'],
                  extra_link_args=['-lllm'] + make_relative_rpath('lib'))

setup(name='mnnllm',
      version='0.1',
      description='mnn-llm python',
      ext_modules=[module],
      packages=packages,
      data_files=lib_files,
      author='wangzhaode',
      author_email='hi@zhaode.wang')