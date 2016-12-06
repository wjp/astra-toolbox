# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://sf.net/projects/astra-toolbox
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import sys
import os, os.path

import subprocess
import shutil
import setuptools
import glob
from distutils.version import LooseVersion
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import Cython
    if LooseVersion(Cython.__version__)<LooseVersion('0.13'): raise ImportError("Cython version should be at least 0.13")
    import numpy as np
except ImportError:
    np = None
    pass

class copy_ext(build_ext):
  def build_extension(self, ext):
    f = self.get_ext_filename(ext.name)
    ff = self.get_ext_fullpath(ext.name)
    subprocess.call(['cp',f,ff])

pkgdata={}
cmdclass = { 'build_ext': copy_ext }
ext_modules = [ ]

if not 'sdist' in sys.argv and not 'egg_info' in sys.argv:
    usecuda=False
    try:
        cuda_root=os.environ['CUDA_ROOT']
        usecuda=True
    except KeyError:
        pass

    # Compile ASTRA C++ library
    savedPath = os.getcwd()

    os.chdir('astra-toolbox/build/linux/')
    subprocess.call(['./autogen.sh'])
    confcall = ['./configure','--with-install-type=module', '--with-python', '--prefix={}/build/'.format(savedPath)]
    if usecuda:
        confcall.append('--with-cuda={}'.format(cuda_root))
    subprocess.call(confcall)

    os.environ['PYTHONPATH'] = savedPath + '/astra'
    makecall = ['make']
    try:
        makeopts = os.environ['MAKEOPTS'].split()
        makecall.extend(makeopts)
    except KeyError:
        pass
    makecall.append('install-python')
    makecall.append('install-libraries')
    subprocess.call(makecall)

    os.chdir(savedPath)

    subprocess.call(['rm', '-rf', 'astra'])
    subprocess.call(['mv build/python/astra astra'],shell=True)

    # copy libastra.so.0
    subprocess.call("mv build/lib/`. '" + os.path.join(savedPath, 'astra-toolbox', 'build', 'linux', 'libastra.la') + "' ; echo $dlname` astra",shell=True)

    pkgdata['astra']=['libastra.so*']

reqpkgs = ["numpy","six","scipy","cython"]

setup (name = 'astra-toolbox',
       version = '1.8',
       description = 'Python interface to the ASTRA Toolbox',
       author='D.M. Pelt',
       author_email='D.M.Pelt@cwi.nl',
       url='http://www.astra-toolbox.com/',
       license='GPLv3',
       cmdclass = cmdclass,
       packages=['astra', 'astra.plugins'],
       ext_modules=[
         Extension("astra.algorithm_c", sources=[]),
         Extension("astra.astra_c", sources=[]),
         Extension("astra.data2d_c", sources=[]),
         Extension("astra.data3d_c", sources=[]),
         Extension("astra.experimental", sources=[]),
         Extension("astra.extrautils", sources=[]),
         Extension("astra.log_c", sources=[]),
         Extension("astra.matrix_c", sources=[]),
         Extension("astra.plugin_c", sources=[]),
         Extension("astra.projector3d_c", sources=[]),
         Extension("astra.projector_c", sources=[]),
         Extension("astra.utils", sources=[]),
       ],
       package_data=pkgdata,
       install_requires=reqpkgs,
       requires=reqpkgs,
	)
