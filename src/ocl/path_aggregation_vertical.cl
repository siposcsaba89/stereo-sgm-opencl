asdasd
// numeric types
#define uint64_t ulong
#define uint32_t uint
#define uint16_t ushort
#define uint8_t uchar

#define int64_t long
#define int32_t int
#define int16_t short
#define int8_t char
// end numeric type defs


// pixel and feature type defs
#define pixel_type uint8_t
#define feature_type uint32_t

// census transfrom defines
#define WINDOW_WIDTH  9
#define WINDOW_HEIGHT  7
#define BLOCK_SIZE_CENSUS 128
#define LINES_PER_BLOCK 16
#define SMEM_BUFFER_SIZE WINDOW_HEIGHT + 1


//#define threads_per_block  16
//
//#define swidth (threads_per_block + HOR)
//#define sheight (threads_per_block + VERT)
//
//
//#define USE_ATOMIC



kernel void census_transform_kernel(
    global feature_type* dest,
    global const pixel_type* src,
    int width,
    int height,
    int pitch)
{
    const int half_kw = WINDOW_WIDTH / 2;
    const int half_kh = WINDOW_HEIGHT / 2;

    local pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE_CENSUS];

    const int tid = get_local_id(0);
    const int x0 = get_group_id(0) * (BLOCK_SIZE_CENSUS - WINDOW_WIDTH + 1) - half_kw;
    const int y0 = get_group_id(1) * LINES_PER_BLOCK;

    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        const int x = x0 + tid, y = y0 - half_kh + i;
        pixel_type value = 0;
        if (0 <= x && x < width && 0 <= y && y < height) {
            value = src[x + y * pitch];
        }
        smem_lines[i][tid] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int i = 0; i < LINES_PER_BLOCK; ++i) {
        if (i + 1 < LINES_PER_BLOCK) {
            // Load to smem
            const int x = x0 + tid, y = y0 + half_kh + i + 1;
            pixel_type value = 0;
            if (0 <= x && x < width && 0 <= y && y < height) {
                value = src[x + y * pitch];
            }
            const int smem_x = tid;
            const int smem_y = (WINDOW_HEIGHT + i) % SMEM_BUFFER_SIZE;
            smem_lines[smem_y][smem_x] = value;
        }

        if (half_kw <= tid && tid < BLOCK_SIZE_CENSUS - half_kw) {
            // Compute and store
            const int x = x0 + tid, y = y0 + i;
            if (half_kw <= x && x < width - half_kw && half_kh <= y && y < height - half_kh) {
                const int smem_x = tid;
                const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
                feature_type f = 0;
                for (int dy = -half_kh; dy < 0; ++dy) {
                    const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
                    const int smem_y2 = (smem_y - dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
                    for (int dx = -half_kw; dx <= half_kw; ++dx) {
                        const int smem_x1 = smem_x + dx;
                        const int smem_x2 = smem_x - dx;
                        const pixel_type a = smem_lines[smem_y1][smem_x1];
                        const pixel_type b = smem_lines[smem_y2][smem_x2];
                        f = (f << 1) | (a > b);
                    }
                }
                for (int dx = -half_kw; dx < 0; ++dx) {
                    const int smem_x1 = smem_x + dx;
                    const int smem_x2 = smem_x - dx;
                    const pixel_type a = smem_lines[smem_y][smem_x1];
                    const pixel_type b = smem_lines[smem_y][smem_x2];
                    f = (f << 1) | (a > b);
                }
                dest[x + y * width] = f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


#define WARP_SIZE 32


#define DP_BLOCK_SIZE_V 16
#define BLOCK_SIZE_V WARP_SIZE * 8
#define MAX_DISPARITY 128

#define SUBGROUP_SIZE_V MAX_DISPARITY / DP_BLOCK_SIZE_V
#define PATHS_PER_WARP_V WARP_SIZE / SUBGROUP_SIZE_V
#define PATHS_PER_BLOCK_V BLOCK_SIZE_V / SUBGROUP_SIZE_V
#define RIGHT_BUFFER_SIZE_V MAX_DISPARITY + PATHS_PER_BLOCK_V
#define RIGHT_BUFFER_ROWS_V RIGHT_BUFFER_SIZE_V / DP_BLOCK_SIZE_V


inline uint32_t pack_uint8x4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) 
{
    uchar4 uint8x4;
    uint8x4.x = (uint8_t)(x);
    uint8x4.y = (uint8_t)(y);
    uint8x4.z = (uint8_t)(z);
    uint8x4.w = (uint8_t)(w);
    return as_uint(uint8x4);
}

inline void store_uint8_vector_16u(global uint8_t* dest,
    const uint32_t* ptr)
{
    uint4 uint32x4;
    uint32x4.x = pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]);
    uint32x4.y = pack_uint8x4(ptr[4], ptr[5], ptr[6], ptr[7]);
    uint32x4.z = pack_uint8x4(ptr[8], ptr[9], ptr[10], ptr[11]);
    uint32x4.w = pack_uint8x4(ptr[12], ptr[13], ptr[14], ptr[15]);
    global uint4* dest_ptr = (global uint4*) dest;
    *dest_ptr = uint32x4;
}


typedef struct
{
    uint32_t last_min;
    local uint32_t * dp_shfl;
    uint32_t dp[DP_BLOCK_SIZE_V];
} DynamicProgramming;

void init(DynamicProgramming * dp, local uint32_t * dp_local)
{
    dp->last_min = 0;
    dp->dp_shfl = dp_local;
    for (unsigned int i = 0; i < DP_BLOCK_SIZE_V; ++i) { dp->dp[i] = 0; }
    dp->dp_shfl[0] = 0;
    dp->dp_shfl[1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
}

void update(DynamicProgramming* dp,
    uint32_t* local_costs,
    uint32_t p1,
    uint32_t p2,
    uint32_t mask,
    local uint32_t shfl_memory[BLOCK_SIZE_V][2])
{
    const unsigned int lane_id = get_local_id(0) % SUBGROUP_SIZE_V;

    //const uint32_t dp0 = dp->dp[0];
    local uint32_t local_min[BLOCK_SIZE_V];
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
    for (unsigned int k = 1; k + 1 < DP_BLOCK_SIZE_V; ++k)
    {
        uint32_t out = min(dp->dp[k] - dp->last_min, p2);
        out = min(out, dp->dp[k - 1] - dp->last_min + p1);
        out = min(out, dp->dp[k + 1] - dp->last_min + p1);
        dp->dp[k - 1] = lazy_out;
        if (k == 1)
        {
            dp->dp_shfl[0] = lazy_out;
        }
        if (k == DP_BLOCK_SIZE_V - 1)
        {
            dp->dp_shfl[1] = lazy_out;
        }
        lazy_out = out + local_costs[k];
        local_min[get_local_id(0)] = min(local_min[get_local_id(0)], lazy_out);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
        const unsigned int k = DP_BLOCK_SIZE_V - 1;
        const int shfl_next_idx = min(BLOCK_SIZE_V - 1, (int)get_local_id(0) + 1);
        const uint32_t next = shfl_memory[shfl_next_idx][0];
//#if CUDA_VERSION >= 9000
//        const uint32_t next = __shfl_down_sync(mask, dp0, 1);
//#else
//        const uint32_t next = __shfl_down(dp0, 1);
//#endif
        uint32_t out = min(dp->dp[k] - dp->last_min, p2);
        out = min(out, dp->dp[k - 1] - dp->last_min + p1);
        if (lane_id + 1 != SUBGROUP_SIZE_V) 
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
    for (int i = SUBGROUP_SIZE_V / 2; i > 0; i >>= 1)
    {
        if (lane_id < i)
        {
            local_min[lid] = min(local_min[lid], local_min[lid + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int sub_group_idx = get_local_id(0) / SUBGROUP_SIZE_V;
    dp->last_min = local_min[sub_group_idx * SUBGROUP_SIZE_V];
    //dp->last_min = subgroup_min<SUBGROUP_SIZE>(local_min, mask);
    barrier(CLK_LOCAL_MEM_FENCE);
}



kernel void aggregate_vertical_path_kernel(
    global uint8_t* dest,
    global const feature_type* left,
    global const feature_type* right,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
//#define DIRECTION 1
    //static_assert(DIRECTION == 1 || DIRECTION == -1, "");
    if (width == 0 || height == 0) {
        return;
    }

    local feature_type right_buffer[2 * DP_BLOCK_SIZE_V][RIGHT_BUFFER_ROWS_V + 1];
    //buffer for shuffle 
    local feature_type shfl_buffer[BLOCK_SIZE_V][2];
    //ocal feature_type shfl_buffer[1][1];

    DynamicProgramming dp;
    init(&dp, shfl_buffer[get_local_id(0)]);
    const unsigned int warp_id = get_local_id(0) / WARP_SIZE;
    const unsigned int group_id = get_local_id(0) % WARP_SIZE / SUBGROUP_SIZE_V;
    const unsigned int lane_id = get_local_id(0) % SUBGROUP_SIZE_V;
    const unsigned int shfl_mask = (1 << SUBGROUP_SIZE_V) - 1u;
        //generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

    const unsigned int x =
        get_group_id(0) * PATHS_PER_BLOCK_V +
        warp_id * PATHS_PER_WARP_V +
        group_id;
    const unsigned int right_x0 = get_group_id(0) * PATHS_PER_BLOCK_V;
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE_V;

    const unsigned int right0_addr =
        (right_x0 + PATHS_PER_BLOCK_V - 1) - x + dp_offset;
    const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE_V;
    const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE_V;

    for (unsigned int iter = 0; iter < height; ++iter)
    {
        const unsigned int y = iter;// (DIRECTION > 0 ? iter : height - 1 - iter);
        // Load left to register
        feature_type left_value;
        if (x < width)
        {
            left_value = left[x + y * width];
        }
        // Load right to smem
        for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE_V; i0 += BLOCK_SIZE_V)
        {
            const unsigned int i = i0 + get_local_id(0);
            if (i < RIGHT_BUFFER_SIZE_V)
            {
                const int right_x = (int)(right_x0 + PATHS_PER_BLOCK_V - 1 - i - min_disp);
                feature_type right_value = 0;
                if (0 <= right_x && right_x < (int)(width)) 
                {
                    right_value = right[right_x + y * width];
                }
                const unsigned int lo = i % DP_BLOCK_SIZE_V;
                const unsigned int hi = i / DP_BLOCK_SIZE_V;
                right_buffer[lo][hi] = right_value;
                if (hi > 0)
                {
                    right_buffer[lo + DP_BLOCK_SIZE_V][hi - 1] = right_value;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compute
        if (x < width)
        {
            feature_type right_values[DP_BLOCK_SIZE_V];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE_V; ++j)
            {
                right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
            }
            uint32_t local_costs[DP_BLOCK_SIZE_V];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE_V; ++j)
            {
                local_costs[j] = popcount(left_value ^ right_values[j]);
            }
            update(&dp, local_costs, p1, p2, shfl_mask, shfl_buffer);
            store_uint8_vector_16u(
                &dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width],
                dp.dp);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


kernel void aggregate_vertical_path_kernel_down2up(
    global uint8_t* dest,
    global const feature_type* left,
    global const feature_type* right,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
    //static_assert(DIRECTION == 1 || DIRECTION == -1, "");
    if (width == 0 || height == 0) {
        return;
    }

    local feature_type right_buffer[2 * DP_BLOCK_SIZE_V][RIGHT_BUFFER_ROWS_V + 1];
    //buffer for shuffle 
    local feature_type shfl_buffer[BLOCK_SIZE_V][2];
    //ocal feature_type shfl_buffer[1][1];

    DynamicProgramming dp;
    init(&dp, shfl_buffer[get_local_id(0)]);
    const unsigned int warp_id = get_local_id(0) / WARP_SIZE;
    const unsigned int group_id = get_local_id(0) % WARP_SIZE / SUBGROUP_SIZE_V;
    const unsigned int lane_id = get_local_id(0) % SUBGROUP_SIZE_V;
    const unsigned int shfl_mask = (1 << SUBGROUP_SIZE_V) - 1u;
    //generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

    const unsigned int x =
        get_group_id(0) * PATHS_PER_BLOCK_V +
        warp_id * PATHS_PER_WARP_V +
        group_id;
    const unsigned int right_x0 = get_group_id(0) * PATHS_PER_BLOCK_V;
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE_V;

    const unsigned int right0_addr =
        (right_x0 + PATHS_PER_BLOCK_V - 1) - x + dp_offset;
    const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE_V;
    const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE_V;

    for (unsigned int iter = 0; iter < height; ++iter)
    {
        const unsigned int y = height - 1 - iter;//(DIRECTION > 0 ? iter : height - 1 - iter);
        // Load left to register
        feature_type left_value;
        if (x < width)
        {
            left_value = left[x + y * width];
        }
        // Load right to smem
        for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE_V; i0 += BLOCK_SIZE_V)
        {
            const unsigned int i = i0 + get_local_id(0);
            if (i < RIGHT_BUFFER_SIZE_V)
            {
                const int right_x = (int)(right_x0 + PATHS_PER_BLOCK_V - 1 - i - min_disp);
                feature_type right_value = 0;
                if (0 <= right_x && right_x < (int)(width))
                {
                    right_value = right[right_x + y * width];
                }
                const unsigned int lo = i % DP_BLOCK_SIZE_V;
                const unsigned int hi = i / DP_BLOCK_SIZE_V;
                right_buffer[lo][hi] = right_value;
                if (hi > 0)
                {
                    right_buffer[lo + DP_BLOCK_SIZE_V][hi - 1] = right_value;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Compute
        if (x < width)
        {
            feature_type right_values[DP_BLOCK_SIZE_V];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE_V; ++j)
            {
                right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
            }
            uint32_t local_costs[DP_BLOCK_SIZE_V];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE_V; ++j)
            {
                local_costs[j] = popcount(left_value ^ right_values[j]);
            }
            update(&dp, local_costs, p1, p2, shfl_mask, shfl_buffer);
            store_uint8_vector_16u(
                &dest[dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width],
                dp.dp);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}





#define DP_BLOCK_SIZE_H 8;
#define DP_BLOCKS_PER_THREAD_H 1;
#define WARPS_PER_BLOCK_H 4;
#define BLOCK_SIZE_H WARP_SIZE * WARPS_PER_BLOCK_H;








//#define MCOST_LINES128 2
//#define DISP_SIZE 128
//#define PATHS_IN_BLOCK 8
//
//#define PENALTY1 20
//#define PENALTY2 100
//#define v_PENALTY1 = (PENALTY1 << 16) | (PENALTY1 << 0);
//#define v_PENALTY2 = (PENALTY2 << 16) | (PENALTY2 << 0);
//
//kernel void matching_cost_kernel_128(
//    global const uint64_t* d_left, global const uint64_t* d_right,
//    global uint8_t* d_cost, int width, int height)
//{
//    int loc_x = get_local_id(0);
//    int loc_y = get_local_id(1);
//    int gr_x = get_group_id(0);
//    //int gr_y = get_group_id(1);
//
//    local uint64_t right_buf[(128 + 128) * MCOST_LINES128];
//    short y = gr_x * MCOST_LINES128 + loc_y;
//    short sh_offset = (128 + 128) * loc_y;
//    { // first 128 pixel
////#pragma unroll
//        //for (short t = 0; t < 128; t += 64) {
//        if (y < height && loc_x < width)
//            right_buf[sh_offset + loc_x] = d_right[y * width + loc_x];
//        else
//            right_buf[sh_offset + loc_x] = 0;
//        //}
//
//        //local uint64_t left_warp_0[32]; 
//        //left_warp_0[loc_x] = d_left[y * width + loc_x];
//        //local uint64_t left_warp_32[32];
//        //left_warp_32[loc_x] = d_left[y * width + loc_x + 32];
//        //local uint64_t left_warp_64[32]; 
//        //left_warp_64[loc_x] = d_left[y * width + loc_x + 64];
//        //local uint64_t left_warp_96[32];
//        //left_warp_96[loc_x] = d_left[y * width + loc_x + 96];
//        //barrier(CLK_LOCAL_MEM_FENCE);
//
//
//#pragma unroll
//        for (short x = 0; x < 128; x++) {
//            if (y < height && x < width)
//            {
//                uint64_t left_val = d_left[y * width + x];// left_warp_0[x];// shfl_u64(left_warp_0, x);
//    //#pragma unroll
//                //for (short k = loc_x; k < DISP_SIZE; k += 64) {
//                uint64_t right_val = x < loc_x ? 0 : right_buf[sh_offset + x - loc_x];
//                int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + loc_x;
//                d_cost[dst_idx] = popcount(left_val ^ right_val);
//            }
//            //}
//        }
//
//        //#pragma unroll
//        //		for (short x = 32; x < 64; x++) {
//        //            uint64_t left_val = d_left[y * width + x];// left_warp_32[x - 32];// shfl_u64(left_warp_32, x);
//        //#pragma unroll
//        //			for (short k = loc_x; k < DISP_SIZE; k += 32) {
//        //				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
//        //				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
//        //				d_cost[dst_idx] = popcount(left_val ^ right_val);
//        //			}
//        //		}
//        //
//        //#pragma unroll
//        //		for (short x = 64; x < 96; x++) {
//        //            uint64_t left_val = d_left[y * width + x];// left_warp_64[x - 64];// shfl_u64(left_warp_64, x);
//        //#pragma unroll
//        //			for (short k = loc_x; k < DISP_SIZE; k += 32) {
//        //				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
//        //				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
//        //				d_cost[dst_idx] = popcount(left_val ^ right_val);
//        //			}
//        //		}
//        //
//        //#pragma unroll
//        //		for (short x = 96; x < 128; x++) {
//        //			uint64_t left_val = d_left[y * width + x];// shfl_u64(left_warp_96, x);
//        //#pragma unroll
//        //			for (short k = loc_x; k < DISP_SIZE; k += 32) {
//        //				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
//        //				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
//        //				d_cost[dst_idx] = popcount(left_val ^ right_val);
//        //			}
//        //		}
//    } // end first 128 pix
//
//
//
//    for (short x = 128; x < width; x += 128) {
//        //if (y < height && x + loc_x < width)
//        {
//
//            //uint64_t left_warp = d_left[y * width + (x + loc_x)];
//            right_buf[sh_offset + loc_x + 128] = d_right[y * width + (x + loc_x)];
//            for (short xoff = 0; xoff < 128; xoff++) {
//                if (y < height && x + xoff < width)
//                {
//                    uint64_t left_val = d_left[y * width + x + xoff];// 0;// shfl_u64(left_warp, xoff);
//        //#pragma unroll
//                    //for (short k = loc_x; k < DISP_SIZE; k += 64) {
//                    uint64_t right_val = right_buf[sh_offset + 128 + xoff - loc_x];
//                    int dst_idx = y * (width * DISP_SIZE) + (x + xoff) * DISP_SIZE + loc_x;
//                    d_cost[dst_idx] = popcount(left_val ^ right_val);
//                }
//                //}
//            }
//            //32 elso elemet kidobjuk
//            right_buf[sh_offset + loc_x + 0] = right_buf[sh_offset + loc_x + 128];
//            //right_buf[sh_offset + loc_x + 64] = right_buf[sh_offset + loc_x + 128];
//            //right_buf[sh_offset + loc_x + 128] = right_buf[sh_offset + loc_x + 96];
//            //right_buf[sh_offset + loc_x + 96] = right_buf[sh_offset + loc_x + 128];
//        }
//    }
//}
//
//
//inline int get_idx_x_0(int width, int j) { return j; }
//inline int get_idx_y_0(int height, int i) { return i; }
//inline int get_idx_x_4(int width, int j) { return width - 1 - j; }
//inline int get_idx_y_4(int height, int i) { return i; }
//inline int get_idx_x_2(int width, int j) { return j; }
//inline int get_idx_y_2(int height, int i) { return i; }
//inline int get_idx_x_6(int width, int j) { return j; }
//inline int get_idx_y_6(int height, int i) { return height - 1 - i; }
//
//
//inline void init_lcost_sh_128(local ushort2* sh) {
//    sh[128 * get_local_id(1) / 2 + get_local_id(0) * 2 + 0] = (ushort2)(0);
//    sh[128 * get_local_id(1) / 2 + get_local_id(0) * 2 + 1] = (ushort2)(0);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    //sh[MAX_ * get_local_id(1) + get_local_id(0) * 4 + 2] = 0;
//    //sh[MAX_ * get_local_id(1) + get_local_id(0) * 4 + 3] = 0;
//}
//
//inline int min_warp(local ushort* minCostNext)
//{
//    int local_index = get_local_id(0) + get_local_id(1) * 32;
//    barrier(CLK_LOCAL_MEM_FENCE);
//    for (int offset = 32 / 2;
//        offset > 0;
//        offset = offset / 2) {
//        if (get_local_id(0) < offset) {
//            ushort other = minCostNext[local_index + offset];
//            ushort mine = minCostNext[local_index];
//            minCostNext[local_index] = (mine < other) ? mine : other;
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    return  minCostNext[get_local_id(1) * 32];
//}
//
//
//inline int min_warp_int(local int* values)
//{
//    int local_index = get_local_id(0) + get_local_id(1) * 32;
//    barrier(CLK_LOCAL_MEM_FENCE);
//    for (int offset = 32 / 2;
//        offset > 0;
//        offset = offset / 2) {
//        if (get_local_id(0) < offset) {
//            int other = values[local_index + offset];
//            int mine = values[local_index];
//            values[local_index] = (mine < other) ? mine : other;
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    return  values[get_local_id(1) * 32];
//}
//
//
//inline int stereo_loop_128(
//    int i, int j, global const uchar4* d_matching_cost,
//    global uint16_t* d_scost, int width, int height, int minCost, local ushort2* lcost_sh,
//    local ushort* minCostNext) {
//
//
//    int idx = i * width + j; // image index
//    int k = get_local_id(0); // (128 disp) k in [0..31]
//    int shIdx = DISP_SIZE * get_local_id(1) / 2 + 2 * k;
//
//    uchar4 diff_tmp = d_matching_cost[idx * DISP_SIZE / 4 + k];
//
//    ushort2 v_diff_L = (ushort2)(diff_tmp.y, diff_tmp.x); // (0x0504) pack( 0x00'[k+1], 0x00'[k+0])
//    ushort2 v_diff_H = (ushort2)(diff_tmp.w, diff_tmp.z); // (0x0706) pack( 0x00'[k+3], 0x00'[k+2])
//
//                                                           // memory layout
//                                                               //              [            this_warp          ]
//                                                               // lcost_sh_prev lcost_sh_curr_L lcost_sh_curr_H lcost_sh_next
//                                                               // -   16bit   -
//
//    ushort2 lcost_sh_curr_L = lcost_sh[shIdx + 0];
//    ushort2 lcost_sh_curr_H = lcost_sh[shIdx + 1];
//    ushort2 lcost_sh_prev, lcost_sh_next;
//
//    if (shIdx + 2 < DISP_SIZE * PATHS_IN_BLOCK / 2)
//        lcost_sh_next = lcost_sh[shIdx + 2];// __shfl_up((int)lcost_sh_curr_H, 1, 32);
//    else
//        lcost_sh_next = lcost_sh_curr_H;
//
//    if (shIdx - 1 > 0)
//        lcost_sh_prev = lcost_sh[shIdx - 1];
//    else
//        lcost_sh_prev = lcost_sh_curr_L;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    ushort2 v_cost0_L = lcost_sh_curr_L;
//    ushort2 v_cost0_H = lcost_sh_curr_H;
//    ushort2 v_cost1_L = (ushort2)(lcost_sh_curr_L.y, lcost_sh_prev.x);// , 0x5432);
//    ushort2 v_cost1_H = (ushort2)(lcost_sh_curr_H.y, lcost_sh_curr_L.x); // 0x5432);
//
//    ushort2 v_cost2_L = (ushort2)(lcost_sh_curr_H.y, lcost_sh_curr_L.x);// 0x5432);
//    ushort2 v_cost2_H = (ushort2)(lcost_sh_next.y, lcost_sh_curr_H.x);//, 0x5432);
//
//    ushort2 v_minCost = (ushort2)(minCost, minCost);//amd_bytealign(minCost, minCost, 0x1010);
//
//    ushort2 v_cost3 = v_minCost + (ushort2)(PENALTY2, PENALTY2);
//
//    v_cost1_L = v_cost1_L + (ushort2)(PENALTY1);
//    v_cost2_L = v_cost2_L + (ushort2)(PENALTY1);
//
//    v_cost1_H = v_cost1_H + (ushort2)(PENALTY1);
//    v_cost2_H = v_cost2_H + (ushort2)(PENALTY1);
//
//    ushort2 v_tmp_a_L = min(v_cost0_L, v_cost1_L);
//    ushort2 v_tmp_a_H = min(v_cost0_H, v_cost1_H);
//
//    ushort2 v_tmp_b_L = min(v_cost2_L, v_cost3);
//    ushort2 v_tmp_b_H = min(v_cost2_H, v_cost3);
//
//    ushort2 cost_tmp_L = v_diff_L + min(v_tmp_a_L, v_tmp_b_L) - v_minCost;
//    ushort2 cost_tmp_H = v_diff_H + min(v_tmp_a_H, v_tmp_b_H) - v_minCost;
//
//    //itt lehet cserelgetni kell (x, y) -- (y, x)
//    d_scost[DISP_SIZE * idx + k * 4 + 0] += cost_tmp_L.y;
//    d_scost[DISP_SIZE * idx + k * 4 + 1] += cost_tmp_L.x;
//    d_scost[DISP_SIZE * idx + k * 4 + 2] += cost_tmp_H.y;
//    d_scost[DISP_SIZE * idx + k * 4 + 3] += cost_tmp_H.x;
//    //uint2 cost_tmp_32x2;
//    //cost_tmp_32x2.x = cost_tmp_L;
//    //cost_tmp_32x2.y = cost_tmp_H;
//    // if no overflow, __vadd2(x,y) == x + y
////#ifdef USE_ATOMIC
////	atomicAdd((unsigned long long int*)dst, *reinterpret_cast<unsigned long long int*>(&cost_tmp_32x2)); // parhztamossag miatt kell szerintem
////#else
////	*dst = *reinterpret_cast<uint64_t*>(&cost_tmp_32x2);
////#endif
//
//    lcost_sh[shIdx + 0] = cost_tmp_L;
//    lcost_sh[shIdx + 1] = cost_tmp_H;
//
//
//    ushort2 cost_tmp = min(cost_tmp_L, cost_tmp_H);
//
//
//
//    minCostNext[get_local_id(1) * 32 + get_local_id(0)] = min(cost_tmp.x, cost_tmp.y);
//
//    return  min_warp(minCostNext);
//}
//
//
//kernel void compute_stereo_horizontal_dir_kernel_0(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//    init_lcost_sh_128(lcost_sh);
//    int i = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    int minCost = 0;
//
//    for (int j = 0; j < width; j++) {
//        minCost = stereo_loop_128(get_idx_y_0(height, i), get_idx_x_0(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//}
//
//kernel void compute_stereo_horizontal_dir_kernel_4(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//    int i = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    int minCost = 0;
//    //#pragma unroll
//    for (int j = 0; j < width; j++) {
//        minCost = stereo_loop_128(get_idx_y_4(height, i), get_idx_x_4(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//}
//
//kernel void compute_stereo_vertical_dir_kernel_2(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//    int j = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    int minCost = 0;
//    //#pragma unroll
//    for (int i = 0; i < height; i++) {
//        minCost = stereo_loop_128(get_idx_y_2(height, i), get_idx_x_2(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//}
//
//
//kernel void compute_stereo_vertical_dir_kernel_6(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//    int j = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    int minCost = 0;
//    //#pragma unroll
//    for (int i = 0; i < height; i++) {
//        minCost = stereo_loop_128(get_idx_y_6(height, i), get_idx_x_6(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//    }
//}
//
//
//
//int get_idx_x_1(int width, int j) { return j; }
//int get_idx_y_1(int height, int i) { return i; }
//int get_idx_x_3(int width, int j) { return width - 1 - j; }
//int get_idx_y_3(int height, int i) { return i; }
//int get_idx_x_5(int width, int j) { return width - 1 - j; }
//int get_idx_y_5(int height, int i) { return height - 1 - i; }
//int get_idx_x_7(int width, int j) { return j; }
//int get_idx_y_7(int height, int i) { return height - 1 - i; }
//
//kernel void compute_stereo_oblique_dir_kernel_1(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//
//    const int num_paths = width + height - 1;
//    int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    if (pathIdx >= num_paths) { return; }
//
//    int i = max(0, -(width - 1) + pathIdx);
//    int j = max(0, width - 1 - pathIdx);
//
//    int minCost = 0;
//
//    //#pragma unroll
//    while (i < height && j < width) {
//        minCost = stereo_loop_128(get_idx_y_1(height, i), get_idx_x_1(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//        i++; j++;
//    }
//}
//
//
//kernel void compute_stereo_oblique_dir_kernel_3(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//
//    const int num_paths = width + height - 1;
//    int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    if (pathIdx >= num_paths) { return; }
//
//    int i = max(0, -(width - 1) + pathIdx);
//    int j = max(0, width - 1 - pathIdx);
//
//    int minCost = 0;
//
//    //#pragma unroll
//    while (i < height && j < width) {
//        minCost = stereo_loop_128(get_idx_y_3(height, i), get_idx_x_3(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//        i++; j++;
//    }
//}
//
//kernel void compute_stereo_oblique_dir_kernel_5(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//
//    const int num_paths = width + height - 1;
//    int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    if (pathIdx >= num_paths) { return; }
//
//    int i = max(0, -(width - 1) + pathIdx);
//    int j = max(0, width - 1 - pathIdx);
//
//    int minCost = 0;
//
//    //#pragma unroll
//    while (i < height && j < width) {
//        minCost = stereo_loop_128(get_idx_y_5(height, i), get_idx_x_5(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//        i++; j++;
//    }
//}
//
//kernel void compute_stereo_oblique_dir_kernel_7(
//    global const uchar4* d_matching_cost, global uint16_t* d_scost, int width, int height)
//{
//    local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
//    local ushort minCostNext[32 * PATHS_IN_BLOCK];
//
//    init_lcost_sh_128(lcost_sh);
//
//    const int num_paths = width + height - 1;
//    int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
//    if (pathIdx >= num_paths) { return; }
//
//    int i = max(0, -(width - 1) + pathIdx);
//    int j = max(0, width - 1 - pathIdx);
//
//    int minCost = 0;
//
//    //#pragma unroll
//    while (i < height && j < width) {
//        minCost = stereo_loop_128(get_idx_y_7(height, i), get_idx_x_7(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
//        //if (i == 345)
//        //	printf("asdasda %d \n", minCost);
//        barrier(CLK_LOCAL_MEM_FENCE);
//        i++; j++;
//    }
//}
//
//
//#define WTA_PIXEL_IN_BLOCK 8
//
//
//kernel void winner_takes_all_kernel128(global ushort* leftDisp, global ushort* rightDisp, global const ushort* d_cost, int width, int height)
//{
//    const float uniqueness = 0.95f;
//
//    int idx = get_local_id(0);
//    int x = get_group_id(0) * WTA_PIXEL_IN_BLOCK + get_local_id(1);
//    int y = get_group_id(1);
//
//    const unsigned cost_offset = DISP_SIZE * (y * width + x);
//    global const ushort* current_cost = d_cost + cost_offset;
//
//    local ushort tmp_costs_block[DISP_SIZE * WTA_PIXEL_IN_BLOCK];
//    local ushort* tmp_costs = tmp_costs_block + DISP_SIZE * get_local_id(1);
//
//    uint32_t tmp_cL1, tmp_cL2; uint32_t tmp_cL3, tmp_cL4;
//    uint32_t tmp_cR1, tmp_cR2; uint32_t tmp_cR3, tmp_cR4;
//
//    // right (1)
//    const int idx_1 = idx * 4 + 0;
//    const int idx_2 = idx * 4 + 1;
//    const int idx_3 = idx * 4 + 2;
//    const int idx_4 = idx * 4 + 3;
//
//    // TODO optimize global memory loads
//    tmp_costs[idx_1] = ((x + (idx_1)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_1)) + idx_1]; // d_cost[y][x + idx0][idx0]
//    tmp_costs[idx_2] = ((x + (idx_2)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_2)) + idx_2];
//    tmp_costs[idx_3] = ((x + (idx_3)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_3)) + idx_3];
//    tmp_costs[idx_4] = ((x + (idx_4)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_4)) + idx_4];
//
//    //tmp_costs[idx_1] = d_cost[DISP_SIZE * (y * width + (x + idx_1)) + idx_1]; // d_cost[y][x + idx0][idx0]
//
//
//    ushort4 tmp_vcL1 = vload4(0, current_cost + idx_1);
//    //const uint2 idx_v = make_uint2((idx_2 << 16) | idx_1, (idx_4 << 16) | idx_3);
//    //ushort4 idx_v = (ushort4)(idx_1, idx_2 , idx_3, idx_4);
//
//    tmp_cR1 = tmp_costs[idx_1];
//    tmp_cR2 = tmp_costs[idx_2];
//    tmp_cR3 = tmp_costs[idx_3];
//    tmp_cR4 = tmp_costs[idx_4];
//
//    tmp_cL1 = (tmp_vcL1.x << 16) + idx_1;// __byte_perm(idx_v.x, tmp_vcL1.x, 0x5410);
//    tmp_cL2 = (tmp_vcL1.y << 16) + idx_2;//__byte_perm(idx_v.x, tmp_vcL1.x, 0x7632);
//    tmp_cL3 = (tmp_vcL1.z << 16) + idx_3; //__byte_perm(idx_v.y, tmp_vcL1.y, 0x5410);
//    tmp_cL4 = (tmp_vcL1.w << 16) + idx_4; //__byte_perm(idx_v.y, tmp_vcL1.y, 0x7632);
//
//    tmp_cR1 = (tmp_cR1 << 16) + idx_1;
//    tmp_cR2 = (tmp_cR2 << 16) + idx_2;
//    tmp_cR3 = (tmp_cR3 << 16) + idx_3;
//    tmp_cR4 = (tmp_cR4 << 16) + idx_4;
//    //////////////////////////////////////
//
//    local int valL1[32 * WTA_PIXEL_IN_BLOCK];
//
//    valL1[idx + get_local_id(1) * 32] = min(min(tmp_cL1, tmp_cL2), min(tmp_cL3, tmp_cL4));
//    int minTempL1 = min_warp_int(valL1);
//
//    int minCostL1 = (minTempL1 >> 16);
//    int minDispL1 = minTempL1 & 0xffff;
//    //////////////////////////////////////
//    if (idx_1 == minDispL1) { tmp_cL1 = 0x7fffffff; }
//    if (idx_2 == minDispL1) { tmp_cL2 = 0x7fffffff; }
//    if (idx_3 == minDispL1) { tmp_cL3 = 0x7fffffff; }
//    if (idx_4 == minDispL1) { tmp_cL4 = 0x7fffffff; }
//
//    valL1[idx + get_local_id(1) * 32] = min(min(tmp_cL1, tmp_cL2), min(tmp_cL3, tmp_cL4));
//    int minTempL2 = min_warp_int(valL1);
//    int minCostL2 = (minTempL2 >> 16);
//    int minDispL2 = minTempL2 & 0xffff;
//    minDispL2 = minDispL2 == 0xffff ? -1 : minDispL2;
//    //////////////////////////////////////
//
//    if (idx_1 + x >= width) { tmp_cR1 = 0x7fffffff; }
//    if (idx_2 + x >= width) { tmp_cR2 = 0x7fffffff; }
//    if (idx_3 + x >= width) { tmp_cR3 = 0x7fffffff; }
//    if (idx_4 + x >= width) { tmp_cR4 = 0x7fffffff; }
//
//    valL1[idx + get_local_id(1) * 32] = min(min(tmp_cR1, tmp_cR2), min(tmp_cR3, tmp_cR4));
//    int minTempR1 = min_warp_int(valL1);
//
//    int minCostR1 = (minTempR1 >> 16);
//    int minDispR1 = minTempR1 & 0xffff;
//    if (minDispR1 == 0xffff) { minDispR1 = -1; }
//    ///////////////////////////////////////////////////////////////////////////////////
//    // right (2)
//    tmp_costs[idx_1] = ((idx_1) == minDispR1 || (x + (idx_1)) >= width) ? 0xffff : tmp_costs[idx_1];
//    tmp_costs[idx_2] = ((idx_2) == minDispR1 || (x + (idx_2)) >= width) ? 0xffff : tmp_costs[idx_2];
//    tmp_costs[idx_3] = ((idx_3) == minDispR1 || (x + (idx_3)) >= width) ? 0xffff : tmp_costs[idx_3];
//    tmp_costs[idx_4] = ((idx_4) == minDispR1 || (x + (idx_4)) >= width) ? 0xffff : tmp_costs[idx_4];
//
//    tmp_cR1 = tmp_costs[idx_1];
//    tmp_cR1 = (tmp_cR1 << 16) + idx_1;
//
//    tmp_cR2 = tmp_costs[idx_2];
//    tmp_cR2 = (tmp_cR2 << 16) + idx_2;
//
//    tmp_cR3 = tmp_costs[idx_3];
//    tmp_cR3 = (tmp_cR3 << 16) + idx_3;
//
//    tmp_cR4 = tmp_costs[idx_4];
//    tmp_cR4 = (tmp_cR4 << 16) + idx_4;
//
//    if (idx_1 + x >= width || idx_1 == minDispR1) { tmp_cR1 = 0x7fffffff; }
//    if (idx_2 + x >= width || idx_2 == minDispR1) { tmp_cR2 = 0x7fffffff; }
//    if (idx_3 + x >= width || idx_3 == minDispR1) { tmp_cR3 = 0x7fffffff; }
//    if (idx_4 + x >= width || idx_4 == minDispR1) { tmp_cR4 = 0x7fffffff; }
//
//    valL1[idx + get_local_id(1) * 32] = min(min(tmp_cR1, tmp_cR2), min(tmp_cR3, tmp_cR4));
//    int minTempR2 = min_warp_int(valL1);
//    int minCostR2 = (minTempR2 >> 16);
//    int minDispR2 = minTempR2 & 0xffff;
//    if (minDispR2 == 0xffff) { minDispR2 = -1; }
//    ///////////////////////////////////////////////////////////////////////////////////
//
//    if (idx == 0) {
//        float lhv = minCostL2 * uniqueness;
//        leftDisp[y * width + x] = (lhv < minCostL1&& abs(minDispL1 - minDispL2) > 1) ? 0 : minDispL1 + 1; // add "+1" 
//        float rhv = minCostR2 * uniqueness;
//        rightDisp[y * width + x] = (rhv < minCostR1&& abs(minDispR1 - minDispR2) > 1) ? 0 : minDispR1 + 1; // add "+1" 
//    }
//}
//
//
//kernel void check_consistency_kernel_left(
//    global ushort* d_leftDisp, global const ushort* d_rightDisp,
//    global const uchar* d_left, int width, int height) {
//
//    const int j = get_global_id(0);
//    const int i = get_global_id(1);
//
//    // left-right consistency check, only on leftDisp, but could be done for rightDisp too
//
//    uchar mask = d_left[i * width + j];
//    int d = d_leftDisp[i * width + j];
//    int k = j - d;
//    if (mask == 0 || d <= 0 || (k >= 0 && k < width && abs(d_rightDisp[i * width + k] - d) > 1)) {
//        // masked or left-right inconsistent pixel -> invalid
//        d_leftDisp[i * width + j] = 0;
//    }
//}
//
//
//// clamp condition
//inline int clampBC(const int x, const int y, const int nx, const int ny)
//{
//    const int idx = clamp(x, 0, nx - 1);
//    const int idy = clamp(y, 0, ny - 1);
//    return idx + idy * nx;
//}
//
//__kernel void median3x3(
//    const __global ushort* restrict input,
//    __global ushort* restrict output,
//    const int nx,
//    const int ny
//)
//{
//    const int idx = get_global_id(0);
//    const int idy = get_global_id(1);
//    const int id = idx + idy * nx;
//
//    if (idx >= nx || idy >= ny)
//        return;
//
//    ushort window[9];
//
//    window[0] = input[clampBC(idx - 1, idy - 1, nx, ny)];
//    window[1] = input[clampBC(idx, idy - 1, nx, ny)];
//    window[2] = input[clampBC(idx + 1, idy - 1, nx, ny)];
//
//    window[3] = input[clampBC(idx - 1, idy, nx, ny)];
//    window[4] = input[clampBC(idx, idy, nx, ny)];
//    window[5] = input[clampBC(idx + 1, idy, nx, ny)];
//
//    window[6] = input[clampBC(idx - 1, idy + 1, nx, ny)];
//    window[7] = input[clampBC(idx, idy + 1, nx, ny)];
//    window[8] = input[clampBC(idx + 1, idy + 1, nx, ny)];
//
//    // perform partial bitonic sort to find current median
//    ushort flMin = min(window[0], window[1]);
//    ushort flMax = max(window[0], window[1]);
//    window[0] = flMin;
//    window[1] = flMax;
//
//    flMin = min(window[3], window[2]);
//    flMax = max(window[3], window[2]);
//    window[3] = flMin;
//    window[2] = flMax;
//
//    flMin = min(window[2], window[0]);
//    flMax = max(window[2], window[0]);
//    window[2] = flMin;
//    window[0] = flMax;
//
//    flMin = min(window[3], window[1]);
//    flMax = max(window[3], window[1]);
//    window[3] = flMin;
//    window[1] = flMax;
//
//    flMin = min(window[1], window[0]);
//    flMax = max(window[1], window[0]);
//    window[1] = flMin;
//    window[0] = flMax;
//
//    flMin = min(window[3], window[2]);
//    flMax = max(window[3], window[2]);
//    window[3] = flMin;
//    window[2] = flMax;
//
//    flMin = min(window[5], window[4]);
//    flMax = max(window[5], window[4]);
//    window[5] = flMin;
//    window[4] = flMax;
//
//    flMin = min(window[7], window[8]);
//    flMax = max(window[7], window[8]);
//    window[7] = flMin;
//    window[8] = flMax;
//
//    flMin = min(window[6], window[8]);
//    flMax = max(window[6], window[8]);
//    window[6] = flMin;
//    window[8] = flMax;
//
//    flMin = min(window[6], window[7]);
//    flMax = max(window[6], window[7]);
//    window[6] = flMin;
//    window[7] = flMax;
//
//    flMin = min(window[4], window[8]);
//    flMax = max(window[4], window[8]);
//    window[4] = flMin;
//    window[8] = flMax;
//
//    flMin = min(window[4], window[6]);
//    flMax = max(window[4], window[6]);
//    window[4] = flMin;
//    window[6] = flMax;
//
//    flMin = min(window[5], window[7]);
//    flMax = max(window[5], window[7]);
//    window[5] = flMin;
//    window[7] = flMax;
//
//    flMin = min(window[4], window[5]);
//    flMax = max(window[4], window[5]);
//    window[4] = flMin;
//    window[5] = flMax;
//
//    flMin = min(window[6], window[7]);
//    flMax = max(window[6], window[7]);
//    window[6] = flMin;
//    window[7] = flMax;
//
//    flMin = min(window[0], window[8]);
//    flMax = max(window[0], window[8]);
//    window[0] = flMin;
//    window[8] = flMax;
//
//    window[4] = max(window[0], window[4]);
//    window[5] = max(window[1], window[5]);
//
//    window[6] = max(window[2], window[6]);
//    window[7] = max(window[3], window[7]);
//
//    window[4] = min(window[4], window[6]);
//    window[5] = min(window[5], window[7]);
//
//    output[id] = min(window[4], window[5]);
//}
//
//
//
//kernel void copy_u8_to_u16(global const uchar* input,
//    global ushort* output)
//{
//    int x = get_global_id(0);
//    output[x] = input[x];
//}
//
kernel void clear_buffer(global float8* buff)
{
    int x = get_global_id(0);
    buff[x] = (float8)0;
}