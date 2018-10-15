#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

/*
 * Here I don't implement that fancy tiling I've implemented
 * in transpose. Sorry, but it just requires to much time and effort :)
 */

__kernel void matrix_multiplication(__global float *a,
                                    __global float *b,
                                    __global float *c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N,
                                    __local float *tile_a,
                                    __local float *tile_b)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    int local_row = get_local_id(1);
    int local_col = get_local_id(0);

    int wg_size = get_local_size(0);

    int elems_to_handle = (K + wg_size - 1) / wg_size;

    float res = 0;

    for (int it = 0; it < elems_to_handle; ++it) {
        int row_a = row;
        int col_a = wg_size * it + local_col;
        tile_a[local_row * wg_size + local_col] = a[row_a * K + col_a];

        int row_b = wg_size * it + local_row;
        int col_b = col;
        tile_b[local_row * wg_size + local_col] = b[row_b * N + col_b];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < wg_size; i++) {
            res += tile_a[local_row * wg_size + i] * tile_b[i * wg_size + local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[row * N + col] = res;
}
