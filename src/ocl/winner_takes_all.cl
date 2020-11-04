//inlcude inttypes
//include utilities
@MAX_DISPARITY@
@NUM_PATHS@
@COMPUTE_SUBPIXEL@
@WARPS_PER_BLOCK@
@BLOCK_SIZE@
@SUBPIXEL_SHIFT@


#define WARP_SIZE 32
#define ACCUMULATION_PER_THREAD 16u
#define REDUCTION_PER_THREAD (MAX_DISPARITY / WARP_SIZE)
#define ACCUMULATION_INTERVAL (ACCUMULATION_PER_THREAD / REDUCTION_PER_THREAD)
#define UNROLL_DEPTH ((REDUCTION_PER_THREAD > ACCUMULATION_INTERVAL) ? REDUCTION_PER_THREAD : ACCUMULATION_INTERVAL)
#define INVALID_DISP (uint16_t)(-1)

inline uint32_t pack_cost_index(uint32_t cost, uint32_t index)
{
    union {
        uint32_t uint32;
        ushort2 uint16x2;
    } u;
    u.uint16x2.x = (uint16_t)(index);
    u.uint16x2.y = (uint16_t)(cost);
    return u.uint32;
}

inline uint32_t unpack_cost(uint32_t packed)
{
    return packed >> 16;
}

inline int unpack_index(uint32_t packed)
{
    return packed & 0xffffu;
}

inline uint32_t compute_disparity_normal(uint32_t disp, uint32_t cost, const local uint16_t* smem )
{
    return disp;
}

inline uint32_t compute_disparity_subpixel(uint32_t disp, uint32_t cost, const local uint16_t* smem)
{
    int subp = disp;
    subp <<= SUBPIXEL_SHIFT;
    if (disp > 0 && disp < MAX_DISPARITY - 1)
    {
        const int left = smem[disp - 1];
        const int right = smem[disp + 1];
        const int numer = left - right;
        const int denom = left - 2 * cost + right;
        subp += ((numer << SUBPIXEL_SHIFT) + denom) / (2 * denom);
    }
    return subp;
}

kernel void winner_takes_all_kernel(
    global uint16_t * left_dest,
    global uint16_t * right_dest,
    const global uint8_t * src,
    int width,
    int height,
    int pitch,
    float uniqueness)
{

    const unsigned int cost_step = MAX_DISPARITY * width * height;
    const unsigned int warp_id = get_local_id(0) / WARP_SIZE;
    const unsigned int lane_id = get_local_id(0) % WARP_SIZE;

    const unsigned int y = get_group_id(0) * WARPS_PER_BLOCK + warp_id;
    src += y * MAX_DISPARITY * width;
    left_dest += y * pitch;
    right_dest += y * pitch;

    if (y >= height)
    {
        return;
    }

    local uint16_t smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY];
    local uint32_t shfl_buffer[BLOCK_SIZE];


    uint32_t right_best[REDUCTION_PER_THREAD];
    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
    {
        right_best[i] = 0xffffffffu;
    }

    for (unsigned int x0 = 0; x0 < width; x0 += UNROLL_DEPTH) 
    {
#pragma unroll
        for (unsigned int x1 = 0; x1 < UNROLL_DEPTH; ++x1)
        {
            if (x1 % ACCUMULATION_INTERVAL == 0)
            {
                const unsigned int k = lane_id * ACCUMULATION_PER_THREAD;
                const unsigned int k_hi = k / MAX_DISPARITY;
                const unsigned int k_lo = k % MAX_DISPARITY;
                const unsigned int x = x0 + x1 + k_hi;
                if (x < width)
                {
                    const unsigned int offset = x * MAX_DISPARITY + k_lo;
                    uint32_t sum[ACCUMULATION_PER_THREAD];
                    for (unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i)
                    {
                        sum[i] = 0;
                    }
                    for (unsigned int p = 0; p < NUM_PATHS; ++p)
                    {
                        uint32_t load_buffer[ACCUMULATION_PER_THREAD];
                        g_load_uint8_vector(load_buffer, ACCUMULATION_PER_THREAD, &src[p * cost_step + offset]);
                        for (unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i)
                        {
                            sum[i] += load_buffer[i];
                        }
                    }
                    store_uint16_vector(&smem_cost_sum[warp_id][k_hi][k_lo], ACCUMULATION_PER_THREAD, sum);
                }
//#if CUDA_VERSION >= 9000
//                __syncwarp();
//#else
//                __threadfence_block();
//#endif
                barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
            }
            const unsigned int x = x0 + x1;
            if (x < width)
            {
                // Load sum of costs
                const unsigned int smem_x = x1 % ACCUMULATION_INTERVAL;
                const unsigned int k0 = lane_id * REDUCTION_PER_THREAD;
                uint32_t local_cost_sum[REDUCTION_PER_THREAD];
                load_uint16_vector(local_cost_sum, REDUCTION_PER_THREAD, &smem_cost_sum[warp_id][smem_x][k0]);
                
                // Pack sum of costs and dispairty
                uint32_t local_packed_cost[REDUCTION_PER_THREAD];
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                {
                    local_packed_cost[i] = pack_cost_index(local_cost_sum[i], k0 + i);
                }
                
                // Update left
                uint32_t best = 0xffffffffu;
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                {
                    best = min(best, local_packed_cost[i]);
                }
                shfl_buffer[get_local_id(0)] = best;
                barrier(CLK_LOCAL_MEM_FENCE);
                best = subgroup_min(lane_id, WARP_SIZE, shfl_buffer);
                // + best = subgroup_min<WARP_SIZE>(best, 0xffffffffu);
                
                // Update right
#pragma unroll
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i) 
                {
                    const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
                    const int p = (int)(((x - k) & ~(MAX_DISPARITY - 1)) + k);
                    const unsigned int d = (unsigned int)(x - p);
                    //load data into shared memory
                    shfl_buffer[get_local_id(0)] = local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD];
                    barrier(CLK_LOCAL_MEM_FENCE);
                    const uint32_t recv = shfl_buffer[warp_id * WARP_SIZE + d / REDUCTION_PER_THREAD];
                    barrier(CLK_LOCAL_MEM_FENCE);
//#if CUDA_VERSION >= 9000
//                    const uint32_t recv = __shfl_sync(0xffffffffu,
//                        local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD],
//                        d / REDUCTION_PER_THREAD,
//                        WARP_SIZE);
//#else
//                    const uint32_t recv = __shfl(
//                        local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD],
//                        d / REDUCTION_PER_THREAD,
//                        WARP_SIZE);
//#endif
                    right_best[i] = min(right_best[i], recv);
                    if (d == MAX_DISPARITY - 1) 
                    {
                        if (0 <= p)
                        {
                            right_dest[p] = compute_disparity_normal(unpack_index(right_best[i]), 0, 0);
                        }
                        right_best[i] = 0xffffffffu;
                    }
                }
                // Resume updating left to avoid execution dependency
                const uint32_t bestCost = unpack_cost(best);
                const int bestDisp = unpack_index(best);
                bool uniq = true;
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i) 
                {
                    const uint32_t x = local_packed_cost[i];
                    const bool uniq1 = unpack_cost(x) * uniqueness >= bestCost;
                    const bool uniq2 = abs(unpack_index(x) - bestDisp) <= 1;
                    uniq &= uniq1 || uniq2;
                }
                shfl_buffer[get_local_id(0)] = uniq ? 1u : 0u;
                barrier(CLK_LOCAL_MEM_FENCE);
                uniq = subgroup_min(lane_id, WARP_SIZE, shfl_buffer) == 1u ? true : false;
                //uniq = subgroup_and<WARP_SIZE>(uniq, 0xffffffffu);
                //uniq = true;
                if (lane_id == 0) 
                {
                    if (uniq)
                    {
                        if (COMPUTE_SUBPIXEL == 1)
                            left_dest[x] = compute_disparity_subpixel(bestDisp, bestCost, smem_cost_sum[warp_id][smem_x]);
                        else
                            left_dest[x] = compute_disparity_normal(bestDisp, bestCost, smem_cost_sum[warp_id][smem_x]);
                    }
                    else
                        left_dest[x] = INVALID_DISP;
                }
            }
        }
    }
    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i) 
    {
        const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
        const int p = (int)(((width - k) & ~(MAX_DISPARITY - 1)) + k);
        if (0 <= p && p < width) 
        {
            right_dest[p] = compute_disparity_normal(unpack_index(right_best[i]), 0, 0);
        }
    }
}
