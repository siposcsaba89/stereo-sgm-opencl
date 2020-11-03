// include inttypes.cl
// include utility.cl
// included common.cl

@DIRECTION@
@MAX_DISPARITY@
@DP_BLOCKS_PER_THREAD@
#define WARP_SIZE 32

#define feature_type uint32_t

#define SUBGROUPS_PER_WARP WARP_SIZE / SUBGROUP_SIZE
#define PATHS_PER_WARP WARP_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE
#define PATHS_PER_BLOCK BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE


kernel void aggregate_horizontal_path_kernel(
    global uint8_t* dest,
    global const feature_type* left,
    global const feature_type* right,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
    if (width == 0 || height == 0)
    {
        return;
    }

    feature_type right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
    local feature_type shfl_buffer[BLOCK_SIZE][2];
    local feature_type shfl_buffer_local[BLOCK_SIZE];

    DynamicProgramming dp[DP_BLOCKS_PER_THREAD];
    //TODO : works until DP_BLOCKS_PER_THREAD is 1
    for (int i = 0; i < DP_BLOCKS_PER_THREAD; ++i)
    {
        init(&dp[i], shfl_buffer[get_local_id(0)]);
    }

    const unsigned int warp_id = get_local_id(0) / WARP_SIZE;
    const unsigned int group_id = get_local_id(0) % WARP_SIZE / SUBGROUP_SIZE;
    const unsigned int lane_id = get_local_id(0) % SUBGROUP_SIZE;
    const unsigned int shfl_mask =
        generate_mask() << (group_id * SUBGROUP_SIZE);

    const unsigned int y0 =
        PATHS_PER_BLOCK * get_group_id(0) +
        PATHS_PER_WARP * warp_id +
        group_id;
    const unsigned int feature_step = SUBGROUPS_PER_WARP * width;
    const unsigned int dest_step = SUBGROUPS_PER_WARP * MAX_DISPARITY * width;
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;
    left += y0 * width;
    right += y0 * width;
    dest += y0 * MAX_DISPARITY * width;

    if (y0 >= height) {
        return;
    }

    if (DIRECTION > 0) {
        for (unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i) {
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
                right_buffer[i][j] = 0;
            }
        }
    }
    else {
        for (unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i) {
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j) {
                const int x = (int)(width - (min_disp + j + dp_offset));
                if (0 <= x && x < (int)(width)) {
                    right_buffer[i][j] = right[i * feature_step + x];
                }
                else {
                    right_buffer[i][j] = 0;
                }
            }
        }
    }

    int x0 = (DIRECTION > 0) ? 0 : (int)((width - 1) & ~(DP_BLOCK_SIZE - 1));
    for (unsigned int iter = 0; iter < width; iter += DP_BLOCK_SIZE) {
        for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i) 
        {
            const unsigned int x = x0 + (DIRECTION > 0 ? i : (DP_BLOCK_SIZE - 1 - i));
            if (x >= width) 
            {
                continue;
            }

            for (unsigned int j = 0; j < DP_BLOCKS_PER_THREAD; ++j) 
            {
                const unsigned int y = y0 + j * SUBGROUPS_PER_WARP;
                if (y >= height) {
                    continue;
                }
                const feature_type left_value = left[j * feature_step + x];
                if (DIRECTION > 0) 
                {
                    shfl_buffer_local[get_local_id(0)] = right_buffer[j][DP_BLOCK_SIZE - 1];
                    //Maybe it is not necessary
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k)
                    {
                        right_buffer[j][k] = right_buffer[j][k - 1];
                    }
                    const int shfl_prev_idx = max(0, (int)get_local_id(0) - 1);
                    right_buffer[j][0] = shfl_buffer_local[shfl_prev_idx];
                    barrier(CLK_LOCAL_MEM_FENCE);
//#if CUDA_VERSION >= 9000
//                    right_buffer[j][0] = __shfl_up_sync(shfl_mask, t, 1, SUBGROUP_SIZE);
//#else
//                    right_buffer[j][0] = __shfl_up(t, 1, SUBGROUP_SIZE);
//#endif
                    if (lane_id == 0 && x >= min_disp)
                    {
                        right_buffer[j][0] =
                            right[j * feature_step + x - min_disp];
                    }
                }
                else {
                    shfl_buffer_local[get_local_id(0)] = right_buffer[j][0];
                    //Maybe it is not necessary
                    barrier(CLK_LOCAL_MEM_FENCE);
                    for (unsigned int k = 1; k < DP_BLOCK_SIZE; ++k) {
                        right_buffer[j][k - 1] = right_buffer[j][k];
                    }
                    //Maybe it is not necessary
                    const int shfl_next_idx = min(BLOCK_SIZE - 1, (int)get_local_id(0) + 1);
                    right_buffer[j][DP_BLOCK_SIZE - 1] = shfl_buffer_local[shfl_next_idx];
                    barrier(CLK_LOCAL_MEM_FENCE);
                    //#if CUDA_VERSION >= 9000
//                    right_buffer[j][DP_BLOCK_SIZE - 1] =
//                        __shfl_down_sync(shfl_mask, t, 1, SUBGROUP_SIZE);
//#else
//                    right_buffer[j][DP_BLOCK_SIZE - 1] = __shfl_down(t, 1, SUBGROUP_SIZE);
//#endif

                    if (lane_id + 1 == SUBGROUP_SIZE) {
                        if (x >= min_disp + dp_offset + DP_BLOCK_SIZE - 1) {
                            right_buffer[j][DP_BLOCK_SIZE - 1] =
                                right[j * feature_step + x - (min_disp + dp_offset + DP_BLOCK_SIZE - 1)];
                        }
                        else {
                            right_buffer[j][DP_BLOCK_SIZE - 1] = 0;
                        }
                    }
                }
                uint32_t local_costs[DP_BLOCK_SIZE];
                for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k) {
                    local_costs[k] = popcount(left_value ^ right_buffer[j][k]);
                }
                update(&dp[j],local_costs, p1, p2, shfl_mask, shfl_buffer);
                store_uint8_vector_8u(
                    &dest[j * dest_step + x * MAX_DISPARITY + dp_offset],
                    dp[j].dp);
            }
        }
        x0 += (int)(DP_BLOCK_SIZE) * DIRECTION;
    }
}



kernel void clear_buffer(global float8* buff)
{
    int x = get_global_id(0);
    buff[x] = (float8)0;
}