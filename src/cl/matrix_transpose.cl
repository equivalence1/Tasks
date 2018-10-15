#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

/*
 * Since We have work groups of 16 x 16 in size max, one might think
 * that it's a good idea to have TILE_SIZE equal to 16 (so work group
 * 16x16 and tile 16x16 perfectly match each other, and each work item
 * only loads 1 element from global memory @a to local @tile, and then
 * 1 element back from local @tile to global @at).
 *
 * Though this architecture seems very natural, it has a significant drawback --
 * this way EVERY element from global arrays @a and @at gets loaded into cache line
 * twice, because typical size of a cache line is 128 bit (= 32 floats), and we only
 * read 16 elements from each row which our work item.
 *
 * Thus it's better to have TILE_SIZE = 32.
 *
 * The problem is that we can only have work groups of size 16x16 max.
 * Hence each work item should handle 4 elements of @a and @at. This is
 * why I divide global_work_size by (wg_size_x / TILE_SIZE).
 *
 * TL;DR;
 *
 * This code is huge, but no matter what work_group sizes we use (e.g. 8x8, 16x16, 32x32),
 * it always reads each element of @a and @at to cache once.
 * Naive implementation (where wg_size = 16x16, TILE_SIZE = 16) requires to read each
 * element of @a to cache twice.
 *
 *      K
 *   +-----+
 *   |     |
 * M |   * | <- row
 *   |     |
 *   +-----+
 *       ^
 *       |
 *      col
 */

#define TILE_SIZE 32

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int M, unsigned int K, __local float *tile)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    int local_row = get_local_id(1);
    int local_col = get_local_id(0);

    int wg_row_id = get_group_id(1);
    int wg_col_id = get_group_id(0);

    int wg_rows_size = get_local_size(1);
    int wg_cols_size = get_local_size(0);
    int wg_size = wg_rows_size * wg_cols_size;

    int local_id = local_row * wg_cols_size + local_col;

    int tile_start_row = wg_row_id * TILE_SIZE;
    int tile_start_col = wg_col_id * TILE_SIZE;

    int elems_to_handle = TILE_SIZE * TILE_SIZE / wg_size;

    // each work item loads from @a to @tile @elems_to_handle elements,
    // all of which located in the same column, one under another.
    // So for work group 16x16, each work item works with elements
    // tile[tile_local_row][tile_local_col]
    // tile[tile_local_row + 1][tile_local_col],
    // tile[tile_local_row + 2][tile_local_col],
    // tile[tile_local_row + 3][tile_local_col].
    //
    // On practice, this actually means that:
    // - if work group size = 16x16, then each warp loads 4 tile rows, one after another
    //   (so in the end we have 4 loaded cache lines for 4 rows => 1 cache line for 1 row => win)
    // - if work group size = 32x32, then each warp loads 1 tile row (=> 1 cache line for 1 row => win)
    for (int it = 0; it < elems_to_handle; ++it) {
        int tile_local_row = local_id / TILE_SIZE * elems_to_handle + it;
        int tile_local_col = local_col + (local_row * wg_cols_size) % TILE_SIZE;

        if (tile_local_row < M && tile_local_col < K) {
            int a_idx = (tile_start_row + tile_local_row) * K + (tile_start_col + tile_local_col);
            tile[tile_local_row * TILE_SIZE + tile_local_col] = a[a_idx];
            // yes, I know, dead code, but it's just too useful for debugging to delete it
            // printf("tile[%d][%d] = a[%d][%d] = %f\n", tile_local_row, tile_local_col, (tile_start_row + tile_local_row), (tile_start_col + tile_local_col), a[a_idx]);
        }

        // This barrier ensures that we are completely done with the current row
        // in this tile and we can continue with the next one.
        // This way we only read 1 cache line for 1 tile line
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    tile_start_col ^= tile_start_row;
    tile_start_row ^= tile_start_col;
    tile_start_col ^= tile_start_row;

    for (int it = 0; it < elems_to_handle; ++it) {
        int tile_local_row = local_id / TILE_SIZE * elems_to_handle + it;
        int tile_local_col = local_col + (local_row * wg_cols_size) % TILE_SIZE;

        if (tile_local_row < M && tile_local_col < K) {
            int at_idx = (tile_start_row + tile_local_row) * M + (tile_start_col + tile_local_col);
            at[at_idx] = tile[tile_local_col * TILE_SIZE + tile_local_row];
            // yes, I know, dead code, but it's just too useful for debugging to delete it
            // printf("at[%d][%d] (%d) = tile[%d][%d] = %f\n", (tile_start_row + tile_local_row), (tile_start_col + tile_local_col), at_idx, tile_local_col, tile_local_row, tile[tile_local_col * TILE_SIZE + tile_local_row]);
        }

        // This barrier ensures that we are completely done with the current row
        // in this tile and we can continue with the next one.
        // This way we only write 1 cache line for 1 tile line
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
