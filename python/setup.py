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
        return [f'-Wl,-rpath,@loader_path/../../../{path},-rpath,$ORIGIN/{path}' ]

lib_suffix = 'so'
if IS_DARWIN:
    lib_suffix = 'dylib'

packages = find_packages()
lib_files = [('lib',
            [f'../build/MNN/libMNN.{lib_suffix}',
             f'../build/MNN/express/libMNN_Express.{lib_suffix}',
             f'../build/libllm.{lib_suffix}'])]

module = Extension('cmnnllm',
                  sources=['./mnnllm.cpp'],
                  include_dirs=['../include', '../MNN/include'],
                  library_dirs=['../build'],
                  extra_compile_args=['-std=c++11'],
                  extra_link_args=['-lllm'] + make_relative_rpath('lib'))

setup(name='mnnllm',
      version='0.1',
      language='c++',
      description='mnn-llm python',
      ext_modules=[module],
      packages=packages,
      data_files=lib_files,
      author='wangzhaode',
      author_email='hi@zhaode.wang')
