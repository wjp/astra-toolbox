# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
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
# -----------------------------------------------------------------------

import astra
import numpy as np

vol_geom = astra.create_vol_geom(128, 128, 128)

angles = np.linspace(0, 2*np.pi, 180,False)
proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 128, 192, angles, 2000, 0)

# Create a simple hollow cube phantom
cube = np.zeros((128,128,128))
cube[17:113,17:113,17:113] = 1
cube[33:97,33:97,33:97] = 0

cfg = astra.astra_dict('cuda3d')
cfg['ProjectionKernel'] = 'line'
cfg['ProjectionGeometry'] = proj_geom
cfg['VolumeGeometry'] = vol_geom
projector_id = astra.projector3d.create(cfg)

# Create projection data from this
proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

# Display a single projection image
import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(proj_data[:,20,:])

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
astra.plugin.register(astra.plugins.SIRTPlugin)
cfg = astra.astra_dict('SIRT-PLUGIN')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id
cfg['ProjectorId'] = projector_id



# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 150)

# Get the result
rec = astra.data3d.get(rec_id)
pylab.figure(2)
pylab.imshow(rec[:,:,65])
pylab.show()


# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
