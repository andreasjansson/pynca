from distutils.core import setup, Extension

nca_module = Extension('nca',
                       sources=['nca.cpp'],
                       include_dirs=['/usr/include/eigen2'])

setup(name='NCA',
      version='0.1',
      description='Neighborhood Components Analysis',
      ext_modules=[nca_module])
