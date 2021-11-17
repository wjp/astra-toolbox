/*
-----------------------------------------------------------------------
Copyright: 2010-2021, imec Vision Lab, University of Antwerp
           2014-2021, CWI, Amsterdam

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

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>

#include <cuda.h>
#include "astra/cuda/3d/util3d.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#include "astra/cuda/3d/dims3d.h"

typedef texture<float, 3, cudaReadModeElementType> texture3D;

static texture3D gT_coneLineVolumeTexture;
static texture3D gT_coneLineProjTexture;


namespace astraCUDA3d {

static const unsigned int g_anglesPerBlock = 4;

// thickness of the slices we're splitting the volume up into
static const unsigned int g_blockSlices = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 16;

static const unsigned g_MaxAngles = 1024;
__constant__ float gC_SrcX[g_MaxAngles];
__constant__ float gC_SrcY[g_MaxAngles];
__constant__ float gC_SrcZ[g_MaxAngles];
__constant__ float gC_DetSX[g_MaxAngles];
__constant__ float gC_DetSY[g_MaxAngles];
__constant__ float gC_DetSZ[g_MaxAngles];
__constant__ float gC_DetUX[g_MaxAngles];
__constant__ float gC_DetUY[g_MaxAngles];
__constant__ float gC_DetUZ[g_MaxAngles];
__constant__ float gC_DetVX[g_MaxAngles];
__constant__ float gC_DetVY[g_MaxAngles];
__constant__ float gC_DetVZ[g_MaxAngles];



static bool bindVolumeDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_coneLineVolumeTexture.addressMode[0] = cudaAddressModeBorder;
	gT_coneLineVolumeTexture.addressMode[1] = cudaAddressModeBorder;
	gT_coneLineVolumeTexture.addressMode[2] = cudaAddressModeBorder;
	gT_coneLineVolumeTexture.filterMode = cudaFilterModeLinear;
	gT_coneLineVolumeTexture.normalized = false;

	cudaBindTextureToArray(gT_coneLineVolumeTexture, array, channelDesc);

	// TODO: error value?

	return true;
}

static bool bindProjDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_coneLineProjTexture.addressMode[0] = cudaAddressModeBorder;
	gT_coneLineProjTexture.addressMode[1] = cudaAddressModeBorder;
	gT_coneLineProjTexture.addressMode[2] = cudaAddressModeBorder;
	gT_coneLineProjTexture.filterMode = cudaFilterModeLinear;
	gT_coneLineProjTexture.normalized = false;

	cudaBindTextureToArray(gT_coneLineProjTexture, array, channelDesc);

	// TODO: error value?

	return true;
}



// x=0, y=1, z=2
struct DIR_X {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return x; }
	__device__ float c1(float x, float y, float z) const { return y; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_coneLineVolumeTexture, f0, f1, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f0; }
	__device__ float y(float f0, float f1, float f2) const { return f1; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
	__device__ int ix(int f0, int f1, int f2) const { return f0; }
	__device__ int iy(int f0, int f1, int f2) const { return f1; }
	__device__ int iz(int f0, int f1, int f2) const { return f2; }
};

// y=0, x=1, z=2
struct DIR_Y {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return y; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_coneLineVolumeTexture, f1, f0, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f0; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
	__device__ int ix(int f0, int f1, int f2) const { return f1; }
	__device__ int iy(int f0, int f1, int f2) const { return f0; }
	__device__ int iz(int f0, int f1, int f2) const { return f2; }
};

// z=0, x=1, y=2
struct DIR_Z {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float c0(float x, float y, float z) const { return z; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return y; }
	__device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_coneLineVolumeTexture, f1, f2, f0); }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f2; }
	__device__ float z(float f0, float f1, float f2) const { return f0; }
	__device__ int ix(int f0, int f1, int f2) const { return f1; }
	__device__ int iy(int f0, int f1, int f2) const { return f2; }
	__device__ int iz(int f0, int f1, int f2) const { return f0; }
};

struct SCALE_CUBE {
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1*a2*a2+1.0f) * fOutputScale; }
};

struct SCALE_NONCUBE {
	float fScale1;
	float fScale2;
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1*fScale1+a2*a2*fScale2+1.0f) * fOutputScale; }
};




template<class COORD, class SCALE>
__global__ void cone_FP_line_t(float* D_projData, unsigned int projPitch,
                          unsigned int startSlice,
                          unsigned int startAngle, unsigned int endAngle,
                          const SDimensions3D dims, SCALE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fSrcX = gC_SrcX[angle];
	const float fSrcY = gC_SrcY[angle];
	const float fSrcZ = gC_SrcZ[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];
	const float fDetUZ = gC_DetUZ[angle];
	const float fDetVX = gC_DetVX[angle];
	const float fDetVY = gC_DetVY[angle];
	const float fDetVZ = gC_DetVZ[angle];
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	if (detectorU >= dims.iProjU)
		return;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* X = startSlice to X = endSlice                   */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   ( 1)       ( 0) */
		/* ray:   (y) = (ay) * T + (by) */
		/*        (z)   (az)       (bz) */

		float a1 = (c.c1(fSrcX,fSrcY,fSrcZ) - c.c1(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		float a2 = (c.c2(fSrcX,fSrcY,fSrcZ) - c.c2(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		const float b1 = c.c1(fSrcX,fSrcY,fSrcZ) - a1 * c.c0(fSrcX,fSrcY,fSrcZ);
		const float b2 = c.c2(fSrcX,fSrcY,fSrcZ) - a2 * c.c0(fSrcX,fSrcY,fSrcZ);

		const float fDistCorr = sc.scale(a1, a2);

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;

		float fCurT = startSlice - 0.5f*c.nSlices(dims);
		// Coordinates
		float f1 = a1 * fCurT + b1;
		float f2 = a2 * fCurT + b2;
		// Fractional index
		f1 += 0.5f*c.nDim1(dims);
		f2 += 0.5f*c.nDim2(dims);
		// Detector index
		f1 = floor(f1);
		f2 = floor(f2);

		float fNext0 = fCurT + 1.0f;

		float fNext1 = f1;
		if (a1 > 0) fNext1 += 1.0f;
		fNext1 = (fNext1 - 0.5f*c.nDim1(dims) - b1) / a1;

		float fNext2 = f2;
		if (a2 > 0) fNext2 += 1.0f;
		fNext2 = (fNext2 - 0.5f*c.nDim2(dims) - b2) / a2;

		// Texture coordinates (at center of detector)
		f1 += 0.5f;
		f2 += 0.5f;

		if (a1 == 0.0f) fNext1 = 1.0f/0.0f;
		if (a2 == 0.0f) fNext2 = 1.0f/0.0f;

		float fStep1 = 1.0f;
		if (a1 < 0) fStep1 = -1.0f;
		float fStep2 = 1.0f;
		if (a2 < 0) fStep2 = -1.0f;
		a1 = fabsf(1.0f / a1);
		a2 = fabsf(1.0f / a2);


		//if (fNext0 < fCurT) { printf("0: %f %f\n", fNext0, fCurT); return; }
		//if (fNext1 < fCurT) { printf("0: %f %f\n", fNext1, fCurT); return; }
		//if (fNext2 < fCurT) { printf("0: %f %f\n", fNext2, fCurT); return; }

		while (fNext0 <= endSlice - 0.5f*c.nSlices(dims))
		{
			float fV = c.tex(f0, f1, f2);
			float fPrevT = fCurT;
			if (fNext0 <= fNext1 && fNext0 <= fNext2) {
				// Step in X dir
				//if (fNext0 >= endSlice - 0.5f*c.nSlices(dims))
				//	break;
				fCurT = fNext0;
				fNext0 += 1.0f;
				f0 += 1.0f;
			} else if (fNext1 <= fNext0 && fNext1 <= fNext2) {
				// Step in Y dir
				fCurT = fNext1;
				fNext1 += a1;
				f1 += fStep1;
			} else {
				// Step in Z dir
				fCurT = fNext2;
				fNext2 += a2;
				f2 += fStep2;
			}
			fVal += (fCurT - fPrevT) * fV;
			//fVal += fV;
		}

		fVal *= fDistCorr;

		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}

template<class COORD, class SCALE>
__global__ void cone_BP_line_t(float* D_volData, unsigned int volPitch,
                          unsigned int startSlice,
                          unsigned int startAngle, unsigned int endAngle,
                          unsigned int angleOffset,
                          const SDimensions3D dims, SCALE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fSrcX = gC_SrcX[angle];
	const float fSrcY = gC_SrcY[angle];
	const float fSrcZ = gC_SrcZ[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];
	const float fDetUZ = gC_DetUZ[angle];
	const float fDetVX = gC_DetVX[angle];
	const float fDetVY = gC_DetVY[angle];
	const float fDetVZ = gC_DetVZ[angle];
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	if (detectorU >= dims.iProjU)
		return;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* X = startSlice to X = endSlice                   */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   ( 1)       ( 0) */
		/* ray:   (y) = (ay) * T + (by) */
		/*        (z)   (az)       (bz) */

		float a1 = (c.c1(fSrcX,fSrcY,fSrcZ) - c.c1(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		float a2 = (c.c2(fSrcX,fSrcY,fSrcZ) - c.c2(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		const float b1 = c.c1(fSrcX,fSrcY,fSrcZ) - a1 * c.c0(fSrcX,fSrcY,fSrcZ);
		const float b2 = c.c2(fSrcX,fSrcY,fSrcZ) - a2 * c.c0(fSrcX,fSrcY,fSrcZ);

		const float fDistCorr = sc.scale(a1, a2);

		float fCurT = startSlice - 0.5f*c.nSlices(dims);
		// Coordinates
		float f1 = a1 * fCurT + b1;
		float f2 = a2 * fCurT + b2;
		// Fractional index
		f1 += 0.5f*c.nDim1(dims);
		f2 += 0.5f*c.nDim2(dims);
		// Voxel index
		f1 = floor(f1);
		f2 = floor(f2);

		float fNext0 = fCurT + 1.0f;

		float fNext1 = f1;
		if (a1 > 0) fNext1 += 1.0f;
		fNext1 = (fNext1 - 0.5f*c.nDim1(dims) - b1) / a1;

		float fNext2 = f2;
		if (a2 > 0) fNext2 += 1.0f;
		fNext2 = (fNext2 - 0.5f*c.nDim2(dims) - b2) / a2;

		// Indices
		int c0 = startSlice;
		int c1 = (int)f1; // already rounded
		int c2 = (int)f2; // already rounded

		int iStep1 = 1;
		if (a1 < 0) iStep1 = -1;
		int iStep2 = 1;
		if (a2 < 0) iStep2 = -1;

		if (a1 == 0.0f) fNext1 = 1.0f/0.0f;
		if (a2 == 0.0f) fNext2 = 1.0f/0.0f;

		a1 = fabsf(1.0f / a1);
		a2 = fabsf(1.0f / a2);

		float fVal = tex3D(gT_coneLineProjTexture, detectorU + 0.5f, angle + angleOffset + 0.5f, detectorV + 0.5f) * fDistCorr;

		while (c0 < endSlice)
		{
			float* addr = 0;
			if (c1 >= 0 && c1 < c.nDim1(dims) && c2 >= 0 && c2 < c.nDim2(dims)) {
				addr = &D_volData[(c.iz(c0,c1,c2)*dims.iVolY +c.iy(c0,c1,c2))*volPitch + c.ix(c0,c1,c2)];
			}
			float fPrevT = fCurT;
			if (fNext0 <= fNext1 && fNext0 <= fNext2) {
				// Step in X dir
				fCurT = fNext0;
				fNext0 += 1.0f;
				c0 += 1;
			} else if (fNext1 <= fNext0 && fNext1 <= fNext2) {
				// Step in Y dir
				fCurT = fNext1;
				fNext1 += a1;
				c1 += iStep1;
			} else {
				// Step in Z dir
				fCurT = fNext2;
				fNext2 += a2;
				c2 += iStep2;
			}
			if (addr) {
				atomicAdd(addr, (fCurT - fPrevT) * fVal );
			}
		}
	}
}



bool ConeLineFP_Array_internal(cudaPitchedPtr D_projData,
                  const SDimensions3D& dims, unsigned int angleCount, const SConeProjection* angles,
                  const SProjectorParams3D& params)
{
	// transfer angles to constant memory
	float* tmp = new float[angleCount];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < angleCount; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, angleCount*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(SrcZ);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetSZ);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);
	TRANSFER_TO_CONSTANT(DetUZ);
	TRANSFER_TO_CONSTANT(DetVX);
	TRANSFER_TO_CONSTANT(DetVY);
	TRANSFER_TO_CONSTANT(DetVZ);

#undef TRANSFER_TO_CONSTANT

	delete[] tmp;

	std::list<cudaStream_t> streams;
	dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles

	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	int blockDirection = 0;

	bool cube = true;
	if (abs(params.fVolScaleX / params.fVolScaleY - 1.0) > 0.00001)
		cube = false;
	if (abs(params.fVolScaleX / params.fVolScaleZ - 1.0) > 0.00001)
		cube = false;

	SCALE_CUBE scube;
	scube.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeX;
	float fS1 = params.fVolScaleY / params.fVolScaleX;
	snoncubeX.fScale1 = fS1 * fS1;
	float fS2 = params.fVolScaleZ / params.fVolScaleX;
	snoncubeX.fScale2 = fS2 * fS2;
	snoncubeX.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeY;
	fS1 = params.fVolScaleX / params.fVolScaleY;
	snoncubeY.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleY;
	snoncubeY.fScale2 = fS2 * fS2;
	snoncubeY.fOutputScale = params.fOutputScale * params.fVolScaleY;

	SCALE_NONCUBE snoncubeZ;
	fS1 = params.fVolScaleX / params.fVolScaleZ;
	snoncubeZ.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleZ;
	snoncubeZ.fScale2 = fS2 * fS2;
	snoncubeZ.fOutputScale = params.fOutputScale * params.fVolScaleZ;


	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= angleCount; ++a) {
		int dir = -1;
		if (a != angleCount) {
			float dX = fabsf(angles[a].fSrcX - (angles[a].fDetSX + dims.iProjU*angles[a].fDetUX*0.5f + dims.iProjV*angles[a].fDetVX*0.5f));
			float dY = fabsf(angles[a].fSrcY - (angles[a].fDetSY + dims.iProjU*angles[a].fDetUY*0.5f + dims.iProjV*angles[a].fDetVY*0.5f));
			float dZ = fabsf(angles[a].fSrcZ - (angles[a].fDetSZ + dims.iProjU*angles[a].fDetUZ*0.5f + dims.iProjV*angles[a].fDetVZ*0.5f));

			if (dX >= dY && dX >= dZ)
				dir = 0;
			else if (dY >= dX && dY >= dZ)
				dir = 1;
			else
				dir = 2;
		}

		if (a == angleCount || dir != blockDirection) {
			// block done

			blockEnd = a;
			if (blockStart != blockEnd) {

				dim3 dimGrid(
				             ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
(blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock);
				// TODO: check if we can't immediately
				//       destroy the stream after use
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);

				// printf("angle block: %d to %d, %d (%dx%d, %dx%d)\n", blockStart, blockEnd, blockDirection, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

				if (blockDirection == 0) {
					for (unsigned int i = 0; i < dims.iVolX; i += g_blockSlices)
						if (cube)
							cone_FP_line_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, scube);
						else
							cone_FP_line_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeX);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (cube)
							cone_FP_line_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, scube);
						else
							cone_FP_line_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeY);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (cube)
							cone_FP_line_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, scube);
						else
							cone_FP_line_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeZ);
				}

			}

			blockDirection = dir;
			blockStart = a;
		}
	}

	bool ok = true;

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter) {
		ok &= checkCuda(cudaStreamSynchronize(*iter), "cone line fp");
		cudaStreamDestroy(*iter);
	}

	// printf("%f\n", toc(t));

	return ok;
}

bool ConeLineBP_Array_internal(cudaPitchedPtr D_volData,
                  const SDimensions3D& dims, unsigned int startAngle, unsigned int angleCount, const SConeProjection* angles,
                  const SProjectorParams3D& params)
{
	angles += startAngle;

	// transfer angles to constant memory
	float* tmp = new float[angleCount];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < angleCount; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, angleCount*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(SrcZ);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetSZ);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);
	TRANSFER_TO_CONSTANT(DetUZ);
	TRANSFER_TO_CONSTANT(DetVX);
	TRANSFER_TO_CONSTANT(DetVY);
	TRANSFER_TO_CONSTANT(DetVZ);

#undef TRANSFER_TO_CONSTANT

	delete[] tmp;

	std::list<cudaStream_t> streams;
	dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles

	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	int blockDirection = 0;

	bool cube = true;
	if (abs(params.fVolScaleX / params.fVolScaleY - 1.0) > 0.00001)
		cube = false;
	if (abs(params.fVolScaleX / params.fVolScaleZ - 1.0) > 0.00001)
		cube = false;

	SCALE_CUBE scube;
	scube.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeX;
	float fS1 = params.fVolScaleY / params.fVolScaleX;
	snoncubeX.fScale1 = fS1 * fS1;
	float fS2 = params.fVolScaleZ / params.fVolScaleX;
	snoncubeX.fScale2 = fS2 * fS2;
	snoncubeX.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeY;
	fS1 = params.fVolScaleX / params.fVolScaleY;
	snoncubeY.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleY;
	snoncubeY.fScale2 = fS2 * fS2;
	snoncubeY.fOutputScale = params.fOutputScale * params.fVolScaleY;

	SCALE_NONCUBE snoncubeZ;
	fS1 = params.fVolScaleX / params.fVolScaleZ;
	snoncubeZ.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleZ;
	snoncubeZ.fScale2 = fS2 * fS2;
	snoncubeZ.fOutputScale = params.fOutputScale * params.fVolScaleZ;


	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= angleCount; ++a) {
		int dir = -1;
		if (a != angleCount) {
			float dX = fabsf(angles[a].fSrcX - (angles[a].fDetSX + dims.iProjU*angles[a].fDetUX*0.5f + dims.iProjV*angles[a].fDetVX*0.5f));
			float dY = fabsf(angles[a].fSrcY - (angles[a].fDetSY + dims.iProjU*angles[a].fDetUY*0.5f + dims.iProjV*angles[a].fDetVY*0.5f));
			float dZ = fabsf(angles[a].fSrcZ - (angles[a].fDetSZ + dims.iProjU*angles[a].fDetUZ*0.5f + dims.iProjV*angles[a].fDetVZ*0.5f));

			if (dX >= dY && dX >= dZ)
				dir = 0;
			else if (dY >= dX && dY >= dZ)
				dir = 1;
			else
				dir = 2;
		}

		if (a == angleCount || dir != blockDirection) {
			// block done

			blockEnd = a;
			if (blockStart != blockEnd) {

				dim3 dimGrid(
				             ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
(blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock);
				// TODO: check if we can't immediately
				//       destroy the stream after use
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);

				// printf("angle block: %d to %d, %d (%dx%d, %dx%d)\n", blockStart, blockEnd, blockDirection, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

				if (blockDirection == 0) {
					for (unsigned int i = 0; i < dims.iVolX; i += g_blockSlices)
						if (cube)
							cone_BP_line_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), i, blockStart, blockEnd, startAngle, dims, scube);
						else
							cone_BP_line_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), i, blockStart, blockEnd, startAngle, dims, snoncubeX);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (cube)
							cone_BP_line_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), i, blockStart, blockEnd, startAngle, dims, scube);
						else
							cone_BP_line_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), i, blockStart, blockEnd, startAngle, dims, snoncubeY);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (cube)
							cone_BP_line_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), i, blockStart, blockEnd, startAngle, dims, scube);
						else
							cone_BP_line_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), i, blockStart, blockEnd, startAngle, dims, snoncubeZ);
				}

			}

			blockDirection = dir;
			blockStart = a;
		}
	}

	bool ok = true;

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter) {
		ok &= checkCuda(cudaStreamSynchronize(*iter), "cone line bp");
		cudaStreamDestroy(*iter);
	}

	// printf("%f\n", toc(t));

	return ok;
}


bool ConeLineFP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SConeProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer volume to array

	cudaArray* cuArray = allocateVolumeArray(dims);
	transferVolumeToArray(D_volumeData, cuArray, dims);
	bindVolumeDataTexture(cuArray);

	bool ret;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		cudaPitchedPtr D_subprojData = D_projData;
		D_subprojData.ptr = (char*)D_projData.ptr + iAngle * D_projData.pitch;

		ret = ConeLineFP_Array_internal(D_subprojData,
		                            dims, iEndAngle - iAngle, angles + iAngle,
		                            params);
		if (!ret)
			break;
	}

	cudaFreeArray(cuArray);

	return ret;
}

bool ConeLineBP_Array(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SConeProjection* angles,
                  const SProjectorParams3D& params)
{
	bindProjDataTexture(D_projArray);

	bool ret = true;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		ret = ConeLineBP_Array_internal(D_volumeData, dims, iAngle, iEndAngle - iAngle, angles, params);
		if (!ret)
			break;
	}

	return ret;
}

bool ConeLineBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SConeProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bool ret = ConeLineBP_Array(D_volumeData, cuArray, dims, angles, params);

	cudaFreeArray(cuArray);

	return ret;
}



}
