/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#include "astra/Float32Data3D.h"
#include <sstream>

#ifdef ASTRA_CUDA
#include "../../cuda/3d/mem3d.h"
#endif

using namespace std;

namespace astra {

//----------------------------------------------------------------------------------------
// Default constructor.
CFloat32Data3D::CFloat32Data3D()
{
	m_bInitialized = false;
#ifdef ASTRA_CUDA
	m_pCustomGPUMemory = 0;
#endif
}

//----------------------------------------------------------------------------------------
// Destructor. 
CFloat32Data3D::~CFloat32Data3D() 
{

}
//----------------------------------------------------------------------------------------

bool CFloat32Data3D::_data3DSizesEqual(const CFloat32Data3D * _pA, const CFloat32Data3D * _pB)
{
	return ((_pA->m_iWidth == _pB->m_iWidth) && (_pA->m_iHeight == _pB->m_iHeight) && (_pA->m_iDepth == _pB->m_iDepth));
}

std::string CFloat32Data3D::description() const
{
	std::stringstream res;
	res << m_iWidth << "x" << m_iHeight << "x" << m_iDepth;
	if (getType() == CFloat32Data3D::PROJECTION) res << " sinogram data \t";
	if (getType() == CFloat32Data3D::VOLUME) res << " volume data \t";
	return res.str();
}
//----------------------------------------------------------------------------------------

#ifdef ASTRA_CUDA
CFloat32ExistingGPUMemory::CFloat32ExistingGPUMemory(unsigned int x_, unsigned int y_, unsigned int z_, unsigned int pitch, float *D_ptr) : x(x_), y(y_), z(z_) {
	hnd = astraCUDA3d::wrapHandle(D_ptr, x, y, z, pitch);
}
bool CFloat32ExistingGPUMemory::allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, astraCUDA3d::Mem3DZeroMode zero) {
	assert(x == this->x);
	assert(y == this->y);
	assert(z == this->z);

	if (zero == astraCUDA3d::INIT_ZERO)
		return astraCUDA3d::zeroGPUMemory(hnd, x, y, z);
	else
		return true;
}
bool CFloat32ExistingGPUMemory::copyToGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
	assert(pos.nx == x);
	assert(pos.ny == y);
	assert(pos.nz == z);
	assert(pos.pitch == x);
	assert(pos.subx == 0);
	assert(pos.suby == 0);
	assert(pos.subnx == x);
	assert(pos.subny == y);

	// These are less necessary than x/y, but allowing access to
	// subvolumes needs an interface change
	assert(pos.subz == 0);
	assert(pos.subnz == z);

	return true;
}
bool CFloat32ExistingGPUMemory::copyFromGPUMemory(const astraCUDA3d::SSubDimensions3D &pos) {
	assert(pos.nx == x);
	assert(pos.ny == y);
	assert(pos.nz == z);
	assert(pos.pitch == x);
	assert(pos.subx == 0);
	assert(pos.suby == 0);
	assert(pos.subnx == x);
	assert(pos.subny == y);

	// These are less necessary than x/y, but allowing access to
	// subvolumes needs an interface change
	assert(pos.subz == 0);
	assert(pos.subnz == z);

	return true;
}
bool CFloat32ExistingGPUMemory::freeGPUMemory() {
	return true;
}
#endif

} // end namespace astra
