#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define unsigned int uint
#define SWAP(a,b) {__local uint* tmp=a; a=b; b=tmp;}

__kernel void count(__global uint* as,
                    __global uint* cs,
                    uint n,
                    uint mask_offset,
                    uint mask_width)
{
  uint gid = get_global_id(0);
  uint mask = ((1UL << mask_width) - 1) << mask_offset;
  uint n_vals = 1 << mask_width;

  if (gid < n) {
    uint x = (as[gid] & mask) >> mask_offset;
    for (uint i = 0; i < n_vals; i++)
      cs[i * n + gid] = x == i ? 1 : 0;
  }
}

/*
 * Почему-то у меня не получается передать этот define как параметр
 * (см объявление ocl::Kernel scan), поэтому просто объявляю его тут.
 */
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

__kernel void reorder(__global uint* as,
                      __global uint* os,
                      __global uint* bs,
                      uint n,
                      uint mask_offset,
                      uint mask_width)
{
  uint gid = get_global_id(0);
  uint mask = ((1UL << mask_width) - 1) << mask_offset;
  uint n_vals = 1 << mask_width;

  if (gid < n) {
    uint x = (as[gid] & mask) >> mask_offset;
    bs[os[x * n + gid] - 1] = as[gid];
  }
}
