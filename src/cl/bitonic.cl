#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define unsigned int uint;

__kernel void bitonic_large(__global float* as,
                            uint n,
                            uint bsize,
                            uint skip_size
)
{
  uint gid = get_global_id(0);
  uint up = gid % (2 * bsize) < bsize;

  if (gid % (2 * skip_size) < skip_size && gid + skip_size < n) {
    float a = as[gid];
    float b = as[gid + skip_size];
    if (up ^ (a <= b)) {
      as[gid] = b;
      as[gid + skip_size] = a;
    }
  }
}

// same as in radix sort
#define BLOCK_SIZE 128

__kernel void bitonic_small(__global float* as,
                            uint n,
                            uint bsize,
                            uint skip_size
)
{
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint up = gid % (2 * bsize) < bsize;

  __local float la[BLOCK_SIZE];

  if (gid < n)
    la[lid] = as[gid];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint s = skip_size; s > 0; s /= 2) {
    if (lid % (2 * s) < s && gid + s < n) {
      float a = la[lid];
      float b = la[lid + s];
      if (up ^ (a <= b)) {
        la[lid] = b;
        la[lid + s] = a;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (gid < n)
    as[gid] = la[lid];
}
