#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define unsigned int uint
#define SWAP(a,b) {__local uint* tmp=a; a=b; b=tmp;}

#define BLOCK_SIZE 128

__kernel void scan(__global uint* as,
                   __global uint* bs,
                   __global uint* cs,
                   uint n,
                   int zero_bs)
{
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint block_size = get_local_size(0);
  uint gr_id = get_group_id(0);

  __local uint local_a[BLOCK_SIZE];
  __local uint local_b[BLOCK_SIZE];
  __local uint* a = local_a;
  __local uint* b = local_b;

  // init locals
  if (gid < n) {
    a[lid] = as[gid];
    if (lid == 0) {
      if (!zero_bs)
        a[0] += bs[gr_id];
    }
  } else
    a[lid] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  // hillis_steele
  for (uint s = 1; s < block_size; s <<= 1) {
    if (lid > (s - 1))
      b[lid] = a[lid] + a[lid - s];
    else
      b[lid] = a[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    SWAP(a, b);
  }

  // move to global
  if (gid < n) {
    cs[gid] = a[lid];
    if (lid == 0)
      bs[gr_id + 1] = a[block_size - 1];
  }
}
