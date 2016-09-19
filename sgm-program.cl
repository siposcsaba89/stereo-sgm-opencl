



kernel void test()
{

}

#define HOR  9
#define VERT  7

#define threads_per_block  16

#define swidth threads_per_block + HOR
#define sheight threads_per_block + VERT

#define uint64_t ulong
#define uint32_t uint
#define uint16_t ushort
#define uint8_t uchar

#define int64_t long
#define int32_t int
#define int16_t short
#define int8_t char

#define USE_ATOMIC

kernel void census_kernel(int hor, int vert, global uchar * d_source, global ulong* d_dest, int width, int height)
{
	const int i = get_global_id(1); //threadIdx.y + blockIdx.y * blockDim.y;
	const int j = get_global_id(0);//threadIdx.x + blockIdx.x * blockDim.x;
	const int offset = j + i * width;

	const int rad_h = HOR / 2;
	const int rad_v = VERT / 2;

	
	local short s_source[swidth*sheight];

	/**
	*                  *- blockDim.x
	*                 /
	*      +---------+---+ -- swidth (blockDim.x+HOR)
	*      |         |   |
	*      |    1    | 2 |
	*      |         |   |
	*      +---------+---+ -- blockDim.y
	*      |    3    | 4 |
	*      +---------+---+ -- sheight (blockDim.y+VERT)
	*/

	// 1. left-top side
	const int ii = /*threadIdx.y + blockIdx.y * blockDim.y*/ i - rad_v;
	const int jj = /*threadIdx.x + blockIdx.x * blockDim.x*/ j - rad_h;
	if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
		//s_source[threadIdx.y*swidth + threadIdx.x] = d_source[ii*width + jj];
		s_source[get_local_id(1)*swidth + get_local_id(0)] = d_source[ii*width + jj];
	}

	// 2. right side
	// 2 * blockDim.x >= swidth
	{
		const int ii = /*threadIdx.y + blockIdx.y * blockDim.y*/ i - rad_v;
		const int jj = /*threadIdx.x + blockIdx.x * blockDim.x*/ j - rad_h + get_local_size(0); //blockDim.x;
		if (get_local_id(0) + get_local_size(0) < swidth && get_local_id(1) < sheight) {
			if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
				s_source[get_local_id(1)*swidth + get_local_id(0) + get_local_size(0)] = d_source[ii*width + jj];
			}
		}
	}

	// 3. bottom side
	// 2 * blockDim.y >= sheight
	{
		const int ii = /*threadIdx.y + blockIdx.y * blockDim.y*/ i - rad_v + get_local_size(1);
		const int jj = /*threadIdx.x + blockIdx.x * blockDim.x*/ j - rad_h;
		if (get_local_id(0) < swidth && get_local_id(1) + get_local_size(1) < sheight) {
			if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
				s_source[(get_local_id(1) + get_local_size(1))*swidth + get_local_id(0)] = d_source[ii*width + jj];
			}
		}
	}

	// 4. right-bottom side
	// 2 * blockDim.x >= swidth && 2 * blockDim.y >= sheight
	{
		const int ii = /*threadIdx.y + blockIdx.y * blockDim.y*/ i - rad_v + get_local_size(1);
		const int jj = /*threadIdx.x + blockIdx.x * blockDim.x*/ j - rad_h + get_local_size(0);
		if (get_local_id(0) + get_local_size(0) < swidth && get_local_id(1) + get_local_size(1) < sheight) {
			if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
				s_source[(get_local_id(1) + get_local_size(1))*swidth + get_local_id(0) + get_local_size(0)] = d_source[ii*width + jj];
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	//__syncthreads();

	// TODO can we remove this condition?
	if (rad_v <= i && i < height - rad_v && rad_h <= j && j < width - rad_h)
	{
		const int ii = get_local_id(1) + rad_v;
		const int jj = get_local_id(0) + rad_h;
		const int soffset = jj + ii * swidth;
		// const SRC_T c = d_source[offset];
		const uchar c = s_source[soffset];
		ulong value = 0;

		uint value1 = 0, value2 = 0;

#pragma unroll
		for (int y = -rad_v; y < 0; y++) {
			for (int x = -rad_h; x <= rad_h; x++) {
				// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
				uchar result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
				value1 <<= 1;
				value1 += result;
			}
		}

		int y = 0;
#pragma unroll
		for (int x = -rad_h; x < 0; x++) {
			// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
			uchar result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
			value1 <<= 1;
			value1 += result;
		}

#pragma unroll
		for (int x = 1; x <= rad_h; x++) {
			// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
			uchar result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
			value2 <<= 1;
			value2 += result;
		}

#pragma unroll
		for (int y = 1; y <= rad_v; y++) {
			for (int x = -rad_h; x <= rad_h; x++) {
				// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
				uchar result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
				value2 <<= 1;
				value2 += result;
			}
		}

		value = (ulong)value2;
		value |= (ulong)value1 << (rad_v * (2 * rad_h + 1) + rad_h);

		d_dest[offset] = value;
	}
}


#define MCOST_LINES128 8
#define DISP_SIZE 128
#define PATHS_IN_BLOCK 8

#define PENALTY1 20
#define PENALTY2 100
#define v_PENALTY1 = (PENALTY1 << 16) | (PENALTY1 << 0);
#define v_PENALTY2 = (PENALTY2 << 16) | (PENALTY2 << 0);

kernel void matching_cost_kernel_128(
	global const uint64_t * d_left, global const uint64_t* d_right,
	global uint8_t* d_cost, int width, int height)
{
	int loc_x = get_local_id(0);
	int loc_y = get_local_id(1);
	int gr_x = get_group_id(0);
	int gr_y = get_group_id(1);

	local uint64_t right_buf[(128 + 32) * MCOST_LINES128];
	int y = gr_x * MCOST_LINES128 + loc_y;
	int sh_offset = (128 + 32) * loc_y;
	{ // first 128 pixel
#pragma unroll
		for (int t = 0; t < 128; t += 32) {
			right_buf[sh_offset + loc_x + t] = d_right[y * width + loc_x + t];
		}

		local uint64_t left_warp_0[32]; 
		left_warp_0[loc_x] = d_left[y * width + loc_x];
		local uint64_t left_warp_32[32];
		left_warp_32[loc_x] = d_left[y * width + loc_x + 32];
		local uint64_t left_warp_64[32]; 
		left_warp_64[loc_x] = d_left[y * width + loc_x + 64];
		local uint64_t left_warp_96[32];
		left_warp_96[loc_x] = d_left[y * width + loc_x + 96];
		barrier(CLK_LOCAL_MEM_FENCE);


#pragma unroll
		for (int x = 0; x < 32; x++) {
            uint64_t left_val = d_left[y * width + x];// left_warp_0[x];// shfl_u64(left_warp_0, x);
#pragma unroll
			for (int k = loc_x; k < DISP_SIZE; k += 32) {
				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			}
		}

#pragma unroll
		for (int x = 32; x < 64; x++) {
            uint64_t left_val = d_left[y * width + x];// left_warp_32[x - 32];// shfl_u64(left_warp_32, x);
#pragma unroll
			for (int k = loc_x; k < DISP_SIZE; k += 32) {
				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			}
		}

#pragma unroll
		for (int x = 64; x < 96; x++) {
            uint64_t left_val = d_left[y * width + x];// left_warp_64[x - 64];// shfl_u64(left_warp_64, x);
#pragma unroll
			for (int k = loc_x; k < DISP_SIZE; k += 32) {
				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			}
		}

#pragma unroll
		for (int x = 96; x < 128; x++) {
			uint64_t left_val = d_left[y * width + x];// shfl_u64(left_warp_96, x);
#pragma unroll
			for (int k = loc_x; k < DISP_SIZE; k += 32) {
				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			}
		}
	} // end first 128 pix



	for (int x = 128; x < width; x += 32) {
		//uint64_t left_warp = d_left[y * width + (x + loc_x)];
		right_buf[sh_offset + loc_x + 128] = d_right[y * width + (x + loc_x)];
		for (int xoff = 0; xoff < 32; xoff++) {
            uint64_t left_val = d_left[y * width + x + xoff];// 0;// shfl_u64(left_warp, xoff);
#pragma unroll
			for (int k = loc_x; k < DISP_SIZE; k += 32) {
				uint64_t right_val = right_buf[sh_offset + 128 + xoff - k];
				int dst_idx = y * (width * DISP_SIZE) + (x + xoff) * DISP_SIZE + k;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			}
		}
        //32 elso elemet kidobjuk
		right_buf[sh_offset + loc_x + 0] = right_buf[sh_offset +  loc_x + 32];
		right_buf[sh_offset + loc_x + 32] = right_buf[sh_offset + loc_x + 64];
		right_buf[sh_offset + loc_x + 64] = right_buf[sh_offset + loc_x + 96];
		right_buf[sh_offset + loc_x + 96] = right_buf[sh_offset + loc_x + 128];
	}
}


inline int get_idx_x_0(int width, int j) { return j; }
inline int get_idx_y_0(int height, int i) { return i; }
inline int get_idx_x_4(int width, int j) { return width - 1 - j; }
inline int get_idx_y_4(int height, int i) { return i; }

inline void init_lcost_sh_128(local ushort2* sh) {
	sh[128 * get_local_id(1) / 2 + get_local_id(0) * 2 + 0] = (ushort2)(0);
	sh[128 * get_local_id(1) / 2 + get_local_id(0) * 2 + 1] = (ushort2)(0);
	//sh[MAX_ * get_local_id(1) + get_local_id(0) * 4 + 2] = 0;
	//sh[MAX_ * get_local_id(1) + get_local_id(0) * 4 + 3] = 0;
}

inline int min_warp(local ushort * minCostNext)
{
    int local_index = get_local_id(0) + get_local_id(1) * 32;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offset = 32 / 2;
        offset > 0;
        offset = offset / 2) {
        if (get_local_id(0) < offset) {
            ushort other = minCostNext[local_index + offset];
            ushort mine = minCostNext[local_index];
            minCostNext[local_index] = (mine < other) ? mine : other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
   
    return  minCostNext[get_local_id(1) * 32];
}

inline int stereo_loop_128(
	int i, int j, global const uchar4 *  d_matching_cost,
	global uint16_t *d_scost, int width, int height, ushort2 minCost, local ushort2 *lcost_sh,
    local ushort * minCostNext) {


	int idx = i * width + j; // image index
    int k = get_local_id(0); // (128 disp) k in [0..31]
	int shIdx = DISP_SIZE * get_local_id(1) / 2 + 2 * k;

	uchar4 diff_tmp = d_matching_cost[idx * DISP_SIZE / 4 + k];

    ushort2 v_diff_L = (ushort2)(diff_tmp.y, diff_tmp.x); // (0x0504) pack( 0x00'[k+1], 0x00'[k+0])
    ushort2 v_diff_H = (ushort2)(diff_tmp.w, diff_tmp.z); // (0x0706) pack( 0x00'[k+3], 0x00'[k+2])

														   // memory layout
															   //              [            this_warp          ]
															   // lcost_sh_prev lcost_sh_curr_L lcost_sh_curr_H lcost_sh_next
															   // -   16bit   -

	ushort2 lcost_sh_curr_L = lcost_sh[shIdx + 0];
	ushort2 lcost_sh_curr_H = lcost_sh[shIdx + 1];
    ushort2 lcost_sh_prev, lcost_sh_next;
    
    if (shIdx + 2 < DISP_SIZE * PATHS_IN_BLOCK / 2 )
        lcost_sh_prev = lcost_sh[shIdx + 2];// __shfl_up((int)lcost_sh_curr_H, 1, 32);
    else
        lcost_sh_prev = lcost_sh_curr_H;
	
    if (shIdx - 1 > 0)
        lcost_sh_next = lcost_sh[shIdx - 1];
    else
        lcost_sh_next = lcost_sh_curr_L;
    barrier(CLK_LOCAL_MEM_FENCE);
    
	ushort2 v_cost0_L = lcost_sh_curr_L;
	ushort2 v_cost0_H = lcost_sh_curr_H;
    ushort2 v_cost1_L = (ushort2)(lcost_sh_curr_L.x, lcost_sh_prev.y);// , 0x5432);
    ushort2 v_cost1_H = (ushort2)(lcost_sh_curr_H.x, lcost_sh_curr_L.y); // 0x5432);
    
    ushort2 v_cost2_L = (ushort2)(lcost_sh_curr_H.x, lcost_sh_curr_L.y);// 0x5432);
    ushort2 v_cost2_H = (ushort2)(lcost_sh_next.x, lcost_sh_curr_H.y);//, 0x5432);

    ushort2 v_minCost = minCost;//amd_bytealign(minCost, minCost, 0x1010);
    
	ushort2 v_cost3 = v_minCost + (ushort2)(PENALTY2, PENALTY2);
    
	v_cost1_L = v_cost1_L + (ushort2)(PENALTY1);
	v_cost2_L = v_cost2_L + (ushort2)(PENALTY1);

	v_cost1_H = v_cost1_H + (ushort2)(PENALTY1);
	v_cost2_H = v_cost2_H + (ushort2)(PENALTY1);
    
	ushort2 v_tmp_a_L = min(v_cost0_L, v_cost1_L);
    ushort2 v_tmp_a_H = min(v_cost0_H, v_cost1_H);
    
	ushort2 v_tmp_b_L = min(v_cost2_L, v_cost3);
	ushort2 v_tmp_b_H = min(v_cost2_H, v_cost3);
    
	ushort2 cost_tmp_L = v_diff_L + min(v_tmp_a_L, v_tmp_b_L) - v_minCost;
	ushort2 cost_tmp_H = v_diff_H + min(v_tmp_a_H, v_tmp_b_H) - v_minCost;
    
    //itt lehet cserelgetni kell (x, y) -- (y, x)
    d_scost[DISP_SIZE * idx + k * 4 + 0] += cost_tmp_L.x;
    d_scost[DISP_SIZE * idx + k * 4 + 1] += cost_tmp_L.y;
    d_scost[DISP_SIZE * idx + k * 4 + 2] += cost_tmp_H.x;
    d_scost[DISP_SIZE * idx + k * 4 + 3] += cost_tmp_H.y;
	//uint2 cost_tmp_32x2;
	//cost_tmp_32x2.x = cost_tmp_L;
	//cost_tmp_32x2.y = cost_tmp_H;
	// if no overflow, __vadd2(x,y) == x + y
//#ifdef USE_ATOMIC
//	atomicAdd((unsigned long long int*)dst, *reinterpret_cast<unsigned long long int*>(&cost_tmp_32x2)); // parhztamossag miatt kell szerintem
//#else
//	*dst = *reinterpret_cast<uint64_t*>(&cost_tmp_32x2);
//#endif

    lcost_sh[shIdx + 0] = cost_tmp_L;
    lcost_sh[shIdx + 1] = cost_tmp_H;


	ushort2 cost_tmp = min(cost_tmp_L, cost_tmp_H);
	
    

	minCostNext[get_local_id(1)* 32  + get_local_id(0) ] = min(cost_tmp.x, cost_tmp.y);
    
    return  min_warp(minCostNext);
}


kernel void compute_stereo_horizontal_dir_kernel_0(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
    local ushort minCostNext[32 * PATHS_IN_BLOCK];
	init_lcost_sh_128(lcost_sh);
	int i = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	ushort2 minCost = (ushort2)0;

    for (int j = 0; j < width; j++) {
		minCost = stereo_loop_128(get_idx_y_0(height, i), get_idx_x_0(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
	}
}
