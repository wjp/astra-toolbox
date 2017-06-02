/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#ifndef _CUDA_PSF3D_H
#define _CUDA_PSF3D_H

namespace astra {
class CProjectionGeometry3D;
}

namespace astraCUDA3d {

// input: D_projData: projection data
//        D_PSF: point spread function. If singlePSF is true,
//               this must be shaped like the projection data, but with only
//               one angle. If singlePSF is false, this must be shaped
//               exactly like the projection data.
//        adjoint: if true, apply the adjoint PSF operator
// output: D_projData: transformed projection data

bool applyPSF(cudaPitchedPtr D_projData, cudaPitchedPtr DF_PSF_Re,
        cudaPitchedPtr DF_PSF_Im,
        const astra::CProjectionGeometry3D *pProjGeom,
        bool adjoint, bool singlePSF);

}


#endif
