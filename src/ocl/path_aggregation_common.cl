@DP_BLOCK_SIZE@
@SUBGROUP_SIZE@
@SIZE@
@BLOCK_SIZE@

typedef struct
{
    uint32_t last_min;
    local uint32_t* dp_shfl;
    uint32_t dp[DP_BLOCK_SIZE];
} DynamicProgramming;

void init(DynamicProgramming* dp, local uint32_t* dp_local)
{
    dp->last_min = 0;
    dp->dp_shfl = dp_local;
    for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i) { dp->dp[i] = 0; }
    dp->dp_shfl[0] = 0;
    dp->dp_shfl[1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
}

void update(DynamicProgramming* dp,
    uint32_t* local_costs,
    uint32_t p1,
    uint32_t p2,
    uint32_t mask,
    local uint32_t shfl_memory[BLOCK_SIZE][2])
{
    const unsigned int lane_id = get_local_id(0) % SUBGROUP_SIZE;

    //const uint32_t dp0 = dp->dp[0];
    local uint32_t local_min[BLOCK_SIZE];
    local_min[get_local_id(0)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint32_t lazy_out = 0;
    {
        const unsigned int k = 0;
        const int shfl_prev_idx = max(0, (int)get_local_id(0) - 1);
        const uint32_t prev = shfl_memory[shfl_prev_idx][1];
        //#if CUDA_VERSION >= 9000
        //        const uint32_t prev =
        //            __shfl_up_sync(mask, dp[DP_BLOCK_SIZE - 1], 1);
        //#else
        //        const uint32_t prev = __shfl_up(dp[DP_BLOCK_SIZE - 1], 1);
        //#endif
        uint32_t out = min(dp->dp[k] - dp->last_min, p2);
        if (lane_id != 0) { out = min(out, prev - dp->last_min + p1); }
        out = min(out, dp->dp[k + 1] - dp->last_min + p1);
        lazy_out = local_min[get_local_id(0)] = out + local_costs[k];
    }
    for (unsigned int k = 1; k + 1 < DP_BLOCK_SIZE; ++k)
    {
        uint32_t out = min(dp->dp[k] - dp->last_min, p2);
        out = min(out, dp->dp[k - 1] - dp->last_min + p1);
        out = min(out, dp->dp[k + 1] - dp->last_min + p1);
        dp->dp[k - 1] = lazy_out;
        if (k == 1)
        {
            dp->dp_shfl[0] = lazy_out;
        }
        if (k == DP_BLOCK_SIZE - 1)
        {
            dp->dp_shfl[1] = lazy_out;
        }
        lazy_out = out + local_costs[k];
        local_min[get_local_id(0)] = min(local_min[get_local_id(0)], lazy_out);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        const unsigned int k = DP_BLOCK_SIZE - 1;
        const int shfl_next_idx = min(BLOCK_SIZE - 1, (int)get_local_id(0) + 1);
        const uint32_t next = shfl_memory[shfl_next_idx][0];
        //#if CUDA_VERSION >= 9000
        //        const uint32_t next = __shfl_down_sync(mask, dp0, 1);
        //#else
        //        const uint32_t next = __shfl_down(dp0, 1);
        //#endif
        uint32_t out = min(dp->dp[k] - dp->last_min, p2);
        out = min(out, dp->dp[k - 1] - dp->last_min + p1);
        if (lane_id + 1 != SUBGROUP_SIZE)
        {
            out = min(out, next - dp->last_min + p1);
        }
        dp->dp[k - 1] = lazy_out;
        dp->dp[k] = out + local_costs[k];
        dp->dp_shfl[1] = dp->dp[k];
        local_min[get_local_id(0)] = min(local_min[get_local_id(0)], dp->dp[k]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //calculating subgroup minimum
    int lid = get_local_id(0);
    for (int i = SUBGROUP_SIZE / 2; i > 0; i >>= 1)
    {
        if (lane_id < i)
        {
            local_min[lid] = min(local_min[lid], local_min[lid + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int sub_group_idx = get_local_id(0) / SUBGROUP_SIZE;
    dp->last_min = local_min[sub_group_idx * SUBGROUP_SIZE];
    //dp->last_min = subgroup_min<SUBGROUP_SIZE>(local_min, mask);
    barrier(CLK_LOCAL_MEM_FENCE);
}


unsigned int generate_mask()
{
    return (unsigned int)((1ull << SIZE) - 1u);
}

