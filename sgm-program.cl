



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
#define PATHS_IN_BLOCK 16

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

inline void init_lcost_sh_128(local uint16_t* sh) {
	sh[128 * get_local_id(1) + get_local_id(0) * 4 + 0] = 0;
	sh[128 * get_local_id(1) + get_local_id(0) * 4 + 1] = 0;
	sh[128 * get_local_id(1) + get_local_id(0) * 4 + 2] = 0;
	sh[128 * get_local_id(1) + get_local_id(0) * 4 + 3] = 0;
}

inline int stereo_loop_128(
	int i, int j, global const uint8_t*  d_matching_cost,
	global uint16_t *d_scost, int width, int height, uint32_t minCost, local uint16_t *lcost_sh) {


	int idx = i * width + j;
    int k = get_local_id(0);// << 2;
	int shIdx = DISP_SIZE * get_local_id(1) + k;

	uint32_t diff_tmp = d_matching_cost[idx * DISP_SIZE + k];
	const uint32_t v_zero = 0;
	uint32_t v_diff_L = amd_bytealign(v_zero, diff_tmp, 0x0504); // pack( 0x00'[k+1], 0x00'[k+0])
	uint32_t v_diff_H = amd_bytealign(v_zero, diff_tmp, 0x0706); // pack( 0x00'[k+3], 0x00'[k+2])

	/*														   // memory layout
															   //              [            this_warp          ]
															   // lcost_sh_prev lcost_sh_curr_L lcost_sh_curr_H lcost_sh_next
															   // -   16bit   -

	uint32_t lcost_sh_curr_L = lcost_sh[shIdx + 0];
	uint32_t lcost_sh_curr_H = lcost_sh[shIdx + 1];
    
    uint32_t lcost_sh_prev = lcost_sh[shIdx + 1];// __shfl_up((int)lcost_sh_curr_H, 1, 32);
	uint32_t lcost_sh_next = lcost_sh[shIdx - 1];//__shfl_down((int)lcost_sh_curr_L, 1, 32);

	uint32_t v_cost0_L = lcost_sh_curr_L;
	uint32_t v_cost0_H = lcost_sh_curr_H;
	uint32_t v_cost1_L = amd_bytealign(lcost_sh_prev, lcost_sh_curr_L, 0x5432);
	uint32_t v_cost1_H = amd_bytealign(lcost_sh_curr_L, lcost_sh_curr_H, 0x5432);

	uint32_t v_cost2_L = amd_bytealign(lcost_sh_curr_L, lcost_sh_curr_H, 0x5432);
	uint32_t v_cost2_H = amd_bytealign(lcost_sh_curr_H, lcost_sh_next, 0x5432);

	uint32_t v_minCost = amd_bytealign(minCost, minCost, 0x1010);
    
	uint32_t v_cost3 = v_minCost + v_PENALTY2;

	v_cost1_L = __vadd2(v_cost1_L, v_PENALTY1);
	v_cost2_L = __vadd2(v_cost2_L, v_PENALTY1);

	v_cost1_H = __vadd2(v_cost1_H, v_PENALTY1);
	v_cost2_H = __vadd2(v_cost2_H, v_PENALTY1);

	uint32_t v_tmp_a_L = __vminu2(v_cost0_L, v_cost1_L);
	uint32_t v_tmp_a_H = __vminu2(v_cost0_H, v_cost1_H);

	uint32_t v_tmp_b_L = __vminu2(v_cost2_L, v_cost3);
	uint32_t v_tmp_b_H = __vminu2(v_cost2_H, v_cost3);

	uint32_t cost_tmp_L = __vsub2(__vadd2(v_diff_L, __vminu2(v_tmp_a_L, v_tmp_b_L)), v_minCost);
	uint32_t cost_tmp_H = __vsub2(__vadd2(v_diff_H, __vminu2(v_tmp_a_H, v_tmp_b_H)), v_minCost);

	uint64_t* dst = reinterpret_cast<uint64_t*>(&d_scost[DISP_SIZE * idx + k]);

	uint2 cost_tmp_32x2;
	cost_tmp_32x2.x = cost_tmp_L;
	cost_tmp_32x2.y = cost_tmp_H;
	// if no overflow, __vadd2(x,y) == x + y
#ifdef USE_ATOMIC
	atomicAdd((unsigned long long int*)dst, *reinterpret_cast<unsigned long long int*>(&cost_tmp_32x2));
#else
	*dst = *reinterpret_cast<uint64_t*>(&cost_tmp_32x2);
#endif

	*reinterpret_cast<uint32_t*>(&lcost_sh[shIdx + 0]) = cost_tmp_L;
	*reinterpret_cast<uint32_t*>(&lcost_sh[shIdx + 2]) = cost_tmp_H;

	uint32_t cost_tmp = __vminu2(cost_tmp_L, cost_tmp_H);
	uint16_t cost_0 = cost_tmp >> 16;
	uint16_t cost_1 = cost_tmp & 0xffff;
	int minCostNext = min(cost_0, cost_1);*/
return 0;// min_warp(minCostNext);
}


kernel void compute_stereo_horizontal_dir_kernel_0(
	global const uint8_t* d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local uint16_t lcost_sh[DISP_SIZE * PATHS_IN_BLOCK];
	init_lcost_sh_128(lcost_sh);
	int i = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	int minCost = 0;
#pragma unroll
	for (int j = 0; j < width; j++) {
		minCost = stereo_loop_128(get_idx_y_0(height, i), get_idx_x_0(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh);
	}
}
