// include inttypes.cl
// include utility.cl
// included common.cl

@X_DIRECTION@
@Y_DIRECTION@
@MAX_DISPARITY@
#define WARP_SIZE 32

#define PATHS_PER_WARP (WARP_SIZE / SUBGROUP_SIZE)
#define PATHS_PER_BLOCK (BLOCK_SIZE / SUBGROUP_SIZE)
#define RIGHT_BUFFER_SIZE  (MAX_DISPARITY + PATHS_PER_BLOCK)
#define RIGHT_BUFFER_ROWS  (RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE)

#define feature_type uint32_t

kernel void aggregate_oblique_path_kernel(
    global uint8_t* dest,
    const global feature_type* left,
    const global feature_type* right,
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

    local feature_type right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS];
    local feature_type shfl_buffer[BLOCK_SIZE];
    DynamicProgramming dp;
    init(&dp, shfl_buffer);

    const unsigned int warp_id = get_local_id(0) / WARP_SIZE;
    const unsigned int group_id = get_local_id(0) % WARP_SIZE / SUBGROUP_SIZE;
    const unsigned int lane_id = get_local_id(0) % SUBGROUP_SIZE;
    const unsigned int shfl_mask =
        generate_mask() << (group_id * SUBGROUP_SIZE);

    const int x0 =
        get_group_id(0) * PATHS_PER_BLOCK +
        warp_id * PATHS_PER_WARP +
        group_id +
        (X_DIRECTION > 0 ? - (int)(height - 1) : 0);
    const int right_x00 =
        get_group_id(0) * PATHS_PER_BLOCK +
        (X_DIRECTION > 0 ? -(int)(height - 1) : 0);
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

    const unsigned int right0_addr =
        (unsigned int)(right_x00 + PATHS_PER_BLOCK - 1 - x0) + dp_offset;
    const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
    const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

    for (unsigned int iter = 0; iter < height; ++iter) 
    {
        const int y = (int)(Y_DIRECTION > 0 ? iter : height - 1 - iter);
        const int x = x0 + (int)(iter) * X_DIRECTION;
        const int right_x0 = right_x00 + (int)(iter) * X_DIRECTION;
        // Load right to smem
        for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE) 
        {
            const unsigned int i = i0 + get_local_id(0);
            if (i < RIGHT_BUFFER_SIZE) 
            {
                const int right_x = (int)(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
                feature_type right_value = 0;
                if (0 <= right_x && right_x < (int)(width)) 
                {
                    right_value = right[right_x + y * width];
                }
                const unsigned int lo = i % DP_BLOCK_SIZE;
                const unsigned int hi = i / DP_BLOCK_SIZE;
                right_buffer[lo][hi] = right_value;
                if (hi > 0) 
                {
                    right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compute
        if (0 <= x && x < (int)(width))
        {
            const feature_type left_value = left[x + y * width];
            feature_type right_values[DP_BLOCK_SIZE];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
            }
            uint32_t local_costs[DP_BLOCK_SIZE];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                local_costs[j] = popcount(left_value ^ right_values[j]);
            }
            update(&dp, local_costs, p1, p2, shfl_mask, shfl_buffer);
            store_uint8_vector(
                &dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width], DP_BLOCK_SIZE,
                dp.dp);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
