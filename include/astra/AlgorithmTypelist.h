/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
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

#ifndef _INC_ASTRA_ALGORITHMTYPELIST
#define _INC_ASTRA_ALGORITHMTYPELIST

#include "Algorithm.h"
#include "TypeList.h"

#include "ArtAlgorithm.h"
#include "SirtAlgorithm.h"
#include "SartAlgorithm.h"
#include "ForwardProjectionAlgorithm.h"
#include "BackProjectionAlgorithm.h"
#include "FilteredBackProjectionAlgorithm.h"
#include "CudaBackProjectionAlgorithm.h"
#include "CudaSartAlgorithm.h"
#include "CudaSirtAlgorithm.h"
#include "CudaCglsAlgorithm.h"
#include "CudaEMAlgorithm.h"
#include "CudaForwardProjectionAlgorithm.h"
#include "CglsAlgorithm.h"
#include "CudaCglsAlgorithm3D.h"
#include "CudaSirtAlgorithm3D.h"
#include "CudaForwardProjectionAlgorithm3D.h"
#include "CudaBackProjectionAlgorithm3D.h"
#include "CudaFDKAlgorithm3D.h"
#include "CudaDartMaskAlgorithm.h"
#include "CudaDartMaskAlgorithm3D.h"
#include "CudaDartSmoothingAlgorithm.h"
#include "CudaDartSmoothingAlgorithm3D.h"
#include "CudaDataOperationAlgorithm.h"
#include "CudaRoiSelectAlgorithm.h"
#include "CudaFilteredBackProjectionAlgorithm.h"

namespace astra {

typedef TypeList<
#ifdef ASTRA_CUDA
			CCudaSartAlgorithm,
			CCudaBackProjectionAlgorithm,
			CCudaDartMaskAlgorithm,
			CCudaDartMaskAlgorithm3D,
			CCudaDartSmoothingAlgorithm,
			CCudaDartSmoothingAlgorithm3D,
			CCudaDataOperationAlgorithm,
			CCudaRoiSelectAlgorithm,
			CCudaSirtAlgorithm,
			CCudaCglsAlgorithm,
			CCudaEMAlgorithm,
			CCudaForwardProjectionAlgorithm,
			CCudaCglsAlgorithm3D,
			CCudaFilteredBackProjectionAlgorithm,
			CCudaFDKAlgorithm3D,
			CCudaSirtAlgorithm3D,
			CCudaForwardProjectionAlgorithm3D,
			CCudaBackProjectionAlgorithm3D,
#endif
			CArtAlgorithm,
			CSartAlgorithm,
			CSirtAlgorithm,
			CCglsAlgorithm,
			CBackProjectionAlgorithm,
			CForwardProjectionAlgorithm,
			CFilteredBackProjectionAlgorithm
	> AlgorithmTypeList;

}

#endif
