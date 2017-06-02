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

#include "util3d.h"
#include "psf3d.h"
#include "cuComplex.h"
#include <cufft.h>
#include <cassert>
#include <cuda.h>
#include "../../include/astra/Logging.h"
#include "../../include/astra/ProjectionGeometry3D.h"

#include "../2d/util.h"

namespace astra {

_AstraExport void getRequiredPSFSize(unsigned int projU, unsigned int projV, unsigned int &PSF_U, unsigned int &PSF_V)
{
	PSF_U = astraCUDA3d::calcNextPowerOfTwo(2 * projU) / 2 + 1;
	PSF_V = astraCUDA3d::calcNextPowerOfTwo(2 * projV);
}

}

namespace astraCUDA3d {

// pcOut *= pcIn
__global__ void devMulComplex(cuComplex* pcOut, float* pcInRe, float* pcInIm, size_t n, float factor, unsigned int pitch)
{
	size_t x = threadIdx.x + blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x);
	if (x >= n) return;
	size_t y = threadIdx.x + pitch * ((gridDim.x * blockIdx.y) + blockIdx.x);

	pcOut[x] = make_cuComplex( factor*(pcOut[x].x * pcInRe[y] - pcOut[x].y * pcInIm[y]),
	                           factor*(pcOut[x].x * pcInIm[y] + pcOut[x].y * pcInRe[y]) );
}

// pcOut *= conj(pcIn)
__global__ void devMulConjComplex(cuComplex* pcOut, float* pcInRe, float* pcInIm, size_t n, float factor, unsigned int pitch)
{
	size_t x = threadIdx.x + blockDim.x * ((gridDim.x * blockIdx.y) + blockIdx.x);
	if (x >= n) return;
	size_t y = threadIdx.x + pitch * ((gridDim.x * blockIdx.y) + blockIdx.x);

	pcOut[x] = make_cuComplex( factor*(pcOut[x].x * pcInRe[y] + pcOut[x].y * pcInIm[y]),
	                           factor*(-pcOut[x].x * pcInIm[y] + pcOut[x].y * pcInRe[y]) );
}



// input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
//output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]

bool applyPSF(cudaPitchedPtr D_projData, cudaPitchedPtr DF_PSF_Re,
        cudaPitchedPtr DF_PSF_Im,
        const astra::CProjectionGeometry3D* pProjGeom,
        bool adjoint, bool singlePSF)
{
	unsigned int iProjU = pProjGeom->getDetectorColCount();
	unsigned int iProjV = pProjGeom->getDetectorRowCount();
	unsigned int iProjAngles = pProjGeom->getProjectionCount();

	unsigned int paddedU = astraCUDA3d::calcNextPowerOfTwo(2 * iProjU);
	unsigned int paddedV = astraCUDA3d::calcNextPowerOfTwo(2 * iProjV);

	assert(DF_PSF_Re.pitch == DF_PSF_Im.pitch);

	cudaExtent paddedExtent = make_cudaExtent(paddedU * sizeof(float), 1, paddedV);

	cudaPitchedPtr D_paddedProjData;
	cudaMalloc3D(&D_paddedProjData, paddedExtent);

	ASTRA_DEBUG("D_projData.pitch = %d, D_paddedProjData.pitch = %d", D_projData.pitch, D_paddedProjData.pitch);

	cudaMemcpy3DParms pPad;
	pPad.dstArray = 0;
	pPad.srcArray = 0;
	pPad.srcPtr = D_projData;
	pPad.dstPtr = D_paddedProjData;
	pPad.dstPos = make_cudaPos(0, 0, 0);

	pPad.extent = make_cudaExtent(iProjU * sizeof(float), 1, iProjV);
	pPad.kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy3DParms pBack;
	pBack.dstArray = 0;
	pBack.srcArray = 0;
	pBack.dstPtr = D_projData;
	pBack.srcPtr = D_paddedProjData;
	pBack.srcPos = make_cudaPos(0, 0, 0);

	pBack.extent = make_cudaExtent(iProjU * sizeof(float), 1, iProjV);
	pBack.kind = cudaMemcpyDeviceToDevice;

	cufftHandle plan, planI;
	int FFTn[2] = { paddedV, paddedU };
	int inembed[2] = { (int)paddedV, (int)(D_paddedProjData.pitch / sizeof(float)) };
	int onembed[2] = { (int)paddedV, paddedU / 2 + 1 };
	int fftSize = onembed[0] * onembed[1];
	
	cufftComplex *DF_projData;
	cudaMalloc((void**)&DF_projData, sizeof(cufftComplex) * fftSize);

	cufftResult err = cufftPlanMany(&plan, 2, FFTn,
	                                inembed, 1, D_paddedProjData.pitch * paddedV / sizeof(float),
	                                onembed, 1, fftSize,
	                                CUFFT_R2C, 1);
	if (err != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("CUFFT Error: Unable to create plan");
		return false;
	}
	err = cufftPlanMany(&planI, 2, FFTn,
	                    onembed, 1, fftSize,
	                    inembed, 1, (D_paddedProjData.pitch * paddedV) / sizeof(float),
	                    CUFFT_C2R, 1);
	if (err != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("CUFFT Error: Unable to create plan");
		return false;
	}
		
	dim3 blockSize(paddedU / 2 + 1);
	dim3 gridSize((paddedV + 255) / 256, 256);

	cudaGetLastError();

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		cudaError err2;

		// copy single projection into padded 2d projection
		err2 = cudaMemset3D(D_paddedProjData, 0, paddedExtent);
		astraCUDA::reportCudaError(err2, "PSF padded memset");

		pPad.srcPos = make_cudaPos(0, i, 0);
		pBack.dstPos = make_cudaPos(0, i, 0);

		err2 = cudaMemcpy3D(&pPad);
		astraCUDA::reportCudaError(err2, "PSF proj memcpy in");

		// FFT
		err = cufftExecR2C(plan, (cufftReal*)D_paddedProjData.ptr, DF_projData);
		if (err != CUFFT_SUCCESS) ASTRA_ERROR("Internal error: PSF FFT failed");

		// pointwise multiplication with PSF
		if (singlePSF) {
#if 1
			if (!adjoint) {
				devMulComplex<<<gridSize, blockSize>>>(DF_projData, (float *)DF_PSF_Re.ptr, (float *)DF_PSF_Im.ptr, fftSize, float(1/double(paddedV*paddedU)), DF_PSF_Re.pitch / sizeof(float));
			} else {
				devMulConjComplex<<<gridSize, blockSize>>>(DF_projData, (float *)DF_PSF_Re.ptr, (float *)DF_PSF_Im.ptr, fftSize, float(1/double(paddedV*paddedU)), DF_PSF_Re.pitch / sizeof(float));
			}
#endif
		} else {
			ASTRA_ERROR("PSF per angle not yet implemented");
			return false;	
		}

		// IFFT
		err = cufftExecC2R(planI, DF_projData, (cufftReal*)D_paddedProjData.ptr);
		if (err != CUFFT_SUCCESS) ASTRA_ERROR("Internal error: PSF IFFT failed");

		// copy projection back to 3D projection data
		err2 = cudaMemcpy3D(&pBack);
		astraCUDA::reportCudaError(err2, "PSF proj memcpy out");

	}

	cufftDestroy(plan);
	cufftDestroy(planI);
	cudaFree(DF_projData);
	cudaFree(D_paddedProjData.ptr);

	return true;
}

}
