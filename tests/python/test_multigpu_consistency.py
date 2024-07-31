import astra
import numpy as np

def print_diff(a, b):
    x = np.abs((a-b).reshape(-1))
    print(np.linalg.norm(x, ord=2), np.max(x), np.max(np.abs(a)), np.max(np.abs(b)))


angles = np.linspace(0, 2*np.pi, 1024, endpoint=False)
vg = astra.create_vol_geom(1024, 1024, 1024)
pg = astra.create_proj_geom('cone', 1.0, 1.0, 1024, 1024, angles, 10000, 0)

projector_id = astra.create_projector('cuda3d', pg, vg)
W = astra.OpTomo(projector_id)

phantom_id = astra.data3d.shepp_logan(vg, returnData=False)

astra.set_gpu_index(0)

pid, projdata_single = astra.create_sino3d_gpu(phantom_id, pg, vg, returnData=True)
astra.data3d.delete(pid)

rec_single = W.reconstruct('FDK_CUDA', projdata_single)

for m in ( 0, 100000000 ):
  for x in ( [0,1], [0,1,2], [0,1,2,3] ):
    astra.set_gpu_index(x, memory=m)

    pid, projdata_multi = astra.create_sino3d_gpu(phantom_id, pg, vg, returnData=True)
    astra.data3d.delete(pid)

    print_diff(projdata_single, projdata_multi)
    assert(np.allclose(projdata_multi, projdata_single, rtol=1e-3, atol=1e-1))

    rec_multi = W.reconstruct('FDK_CUDA', projdata_single)

    print_diff(rec_single, rec_multi)
    assert(np.allclose(rec_multi, rec_single, rtol=1e-3, atol=1e-3))

