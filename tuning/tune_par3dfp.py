import kernel_tuner
import logging

kernel_string = """
#define ASTRA_CUDA
#define ASTRA_BUILDING_HIP

#include "astra/cuda/gpu_runtime_wrapper.h"
"""

with open('../cuda/3d/par3d_fp.cu') as f:
    kernel_string += f.read()

with open('../cuda/3d/util3d.cu') as f:
    kernel_string += f.read()

with open('../cuda/2d/util.cu') as f:
    kernel_string += f.read()

with open('../src/Logging.cpp') as f:
    kernel_string += f.read()

with open('../src/Utilities.cpp') as f:
    kernel_string += f.read()

kernel_string += """

#include <sys/time.h>


class CTimer {
public:
    void start() { gettimeofday(&t_init, 0); }
    void end() { timeval ts; gettimeofday(&ts, 0); td = 1000.*(ts.tv_sec - t_init.tv_sec) + 0.001*(ts.tv_usec - t_init.tv_usec); }
    float get() { return (float)td; }

private:
    timeval t_init;
    double td;
};

using namespace astraCUDA3d;

std::vector<SPar3DProjection> genPar3DProjections(unsigned int iProjAngles,
                                      unsigned int iProjU,
                                      unsigned int iProjV,
                                      double fDetUSize,
                                      double fDetVSize,
                                      const float *pfAngles)
{
        SPar3DProjection base;
        base.fRayX = 0.0f;
        base.fRayY = 1.0f;
        base.fRayZ = 0.0f;

        base.fDetSX = iProjU * fDetUSize * -0.5f;
        base.fDetSY = 0.0f;
        base.fDetSZ = iProjV * fDetVSize * -0.5f;

        base.fDetUX = fDetUSize;
        base.fDetUY = 0.0f;
        base.fDetUZ = 0.0f;

        base.fDetVX = 0.0f;
        base.fDetVY = 0.0f;
        base.fDetVZ = fDetVSize;

        std::vector<SPar3DProjection> p;
        p.resize(iProjAngles);

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); p[i].f##name##Z = base.f##name##Z; } while(0)

        for (unsigned int i = 0; i < iProjAngles; ++i) {
                ROTATE0(Ray, i, pfAngles[i]);
                ROTATE0(DetS, i, pfAngles[i]);
                ROTATE0(DetU, i, pfAngles[i]);
                ROTATE0(DetV, i, pfAngles[i]);
        }

#undef ROTATE0

        return p;
}



extern "C"
float tunable()
{
        SDimensions3D dims;
        SProjectorParams3D params;

        dims.iVolX = 1024;
        dims.iVolY = 1024;
        dims.iVolZ = 256;
        dims.iProjAngles = 1024;
        dims.iProjU = 1024;
        dims.iProjV = 256;

        cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
        cudaPitchedPtr D_projData = allocateProjectionData(dims);

        std::vector<float> angles(1024);
        for (int i = 0; i < 1024; ++i) {
            angles[i] = i * (M_PI)/1024;
            //angles[i] = i * (0.5*M_PI)/1024;
            //angles[i] = (i * (0.25*M_PI) + 0.5*M_PI)/1024;
        }

        std::vector<SPar3DProjection> projs = genPar3DProjections(1024, 1024, 256, 1.0, 1.0, &angles[0]);

        CTimer t;

        t.start();

        Par3DFP(D_volumeData, D_projData, dims, &projs[0], params);
        t.end();

        cudaFree(D_volumeData.ptr);
        cudaFree(D_projData.ptr);

        return t.get();
}
"""

kernel_tuner.tune_kernel("tunable", kernel_string, (512,512), [], {"block_size_x": [8, 16, 32], "block_size_y": [4, 8, 16], "block_slices": [4, 8, 16, 32], "det_block_v": [32]}, verbose=True, lang="C", compiler="hipcc", compiler_options=["-I../include"])
