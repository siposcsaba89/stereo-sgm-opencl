



kernel void test()
{

}

#define HOR  9
#define VERT  7

#define threads_per_block  16

#define swidth (threads_per_block + HOR)
#define sheight (threads_per_block + VERT)

#define uint64_t ulong
#define uint32_t uint
#define uint16_t ushort
#define uint8_t uchar

#define int64_t long
#define int32_t int
#define int16_t short
#define int8_t char

#define USE_ATOMIC

kernel void census_kernel(global const uchar * d_source, global ulong* d_dest, int width, int height)
{
	const int i = get_global_id(1); //threadIdx.y + blockIdx.y * blockDim.y;
	const int j = get_global_id(0);//threadIdx.x + blockIdx.x * blockDim.x;
	const int offset = j + i * width;

	const int rad_h = HOR / 2;
	const int rad_v = VERT / 2;

	
	local uchar s_source[swidth*sheight];

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


#define MCOST_LINES128 2
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
	//int gr_y = get_group_id(1);

	local uint64_t right_buf[(128 + 128) * MCOST_LINES128];
	short y = gr_x * MCOST_LINES128 + loc_y;
	short sh_offset = (128 + 128) * loc_y;
	{ // first 128 pixel
//#pragma unroll
		//for (short t = 0; t < 128; t += 64) {
			right_buf[sh_offset + loc_x] = d_right[y * width + loc_x];
		//}

		//local uint64_t left_warp_0[32]; 
		//left_warp_0[loc_x] = d_left[y * width + loc_x];
		//local uint64_t left_warp_32[32];
		//left_warp_32[loc_x] = d_left[y * width + loc_x + 32];
		//local uint64_t left_warp_64[32]; 
		//left_warp_64[loc_x] = d_left[y * width + loc_x + 64];
		//local uint64_t left_warp_96[32];
		//left_warp_96[loc_x] = d_left[y * width + loc_x + 96];
		//barrier(CLK_LOCAL_MEM_FENCE);


#pragma unroll
		for (short x = 0; x < 128; x++) {
            uint64_t left_val = d_left[y * width + x];// left_warp_0[x];// shfl_u64(left_warp_0, x);
//#pragma unroll
			//for (short k = loc_x; k < DISP_SIZE; k += 64) {
				uint64_t right_val = x < loc_x ? 0 : right_buf[sh_offset + x - loc_x];
				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + loc_x;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			//}
		}

//#pragma unroll
//		for (short x = 32; x < 64; x++) {
//            uint64_t left_val = d_left[y * width + x];// left_warp_32[x - 32];// shfl_u64(left_warp_32, x);
//#pragma unroll
//			for (short k = loc_x; k < DISP_SIZE; k += 32) {
//				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
//				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
//				d_cost[dst_idx] = popcount(left_val ^ right_val);
//			}
//		}
//
//#pragma unroll
//		for (short x = 64; x < 96; x++) {
//            uint64_t left_val = d_left[y * width + x];// left_warp_64[x - 64];// shfl_u64(left_warp_64, x);
//#pragma unroll
//			for (short k = loc_x; k < DISP_SIZE; k += 32) {
//				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
//				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
//				d_cost[dst_idx] = popcount(left_val ^ right_val);
//			}
//		}
//
//#pragma unroll
//		for (short x = 96; x < 128; x++) {
//			uint64_t left_val = d_left[y * width + x];// shfl_u64(left_warp_96, x);
//#pragma unroll
//			for (short k = loc_x; k < DISP_SIZE; k += 32) {
//				uint64_t right_val = x < k ? 0 : right_buf[sh_offset + x - k];
//				int dst_idx = y * (width * DISP_SIZE) + x * DISP_SIZE + k;
//				d_cost[dst_idx] = popcount(left_val ^ right_val);
//			}
//		}
	} // end first 128 pix



	for (short x = 128; x < width; x += 128) {
		//uint64_t left_warp = d_left[y * width + (x + loc_x)];
		right_buf[sh_offset + loc_x + 128] = d_right[y * width + (x + loc_x)];
		for (short xoff = 0; xoff < 128; xoff++) {
            uint64_t left_val = d_left[y * width + x + xoff];// 0;// shfl_u64(left_warp, xoff);
//#pragma unroll
			//for (short k = loc_x; k < DISP_SIZE; k += 64) {
				uint64_t right_val = right_buf[sh_offset + 128 + xoff - loc_x];
				int dst_idx = y * (width * DISP_SIZE) + (x + xoff) * DISP_SIZE + loc_x;
				d_cost[dst_idx] = popcount(left_val ^ right_val);
			//}
		}
        //32 elso elemet kidobjuk
		right_buf[sh_offset + loc_x + 0] = right_buf[sh_offset +  loc_x + 128];
		//right_buf[sh_offset + loc_x + 64] = right_buf[sh_offset + loc_x + 128];
		//right_buf[sh_offset + loc_x + 128] = right_buf[sh_offset + loc_x + 96];
		//right_buf[sh_offset + loc_x + 96] = right_buf[sh_offset + loc_x + 128];
	}
}


inline int get_idx_x_0(int width, int j) { return j; }
inline int get_idx_y_0(int height, int i) { return i; }
inline int get_idx_x_4(int width, int j) { return width - 1 - j; }
inline int get_idx_y_4(int height, int i) { return i; }
inline int get_idx_x_2(int width, int j) { return j; }
inline int get_idx_y_2(int height, int i) { return i; }
inline int get_idx_x_6(int width, int j) { return j; }
inline int get_idx_y_6(int height, int i) { return height - 1 - i; }


inline void init_lcost_sh_128(local ushort2* sh) {
	sh[128 * get_local_id(1) / 2 + get_local_id(0) * 2 + 0] = (ushort2)(0);
	sh[128 * get_local_id(1) / 2 + get_local_id(0) * 2 + 1] = (ushort2)(0);
	barrier(CLK_LOCAL_MEM_FENCE);
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


inline int min_warp_int(local int * values)
{
	int local_index = get_local_id(0) + get_local_id(1) * 32;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = 32 / 2;
		offset > 0;
		offset = offset / 2) {
		if (get_local_id(0) < offset) {
			int other = values[local_index + offset];
			int mine = values[local_index];
			values[local_index] = (mine < other) ? mine : other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	return  values[get_local_id(1) * 32];
}


inline int stereo_loop_128(
	int i, int j, global const uchar4 *  d_matching_cost,
	global uint16_t *d_scost, int width, int height, int minCost, local ushort2 *lcost_sh,
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
		lcost_sh_next = lcost_sh[shIdx + 2];// __shfl_up((int)lcost_sh_curr_H, 1, 32);
    else
		lcost_sh_next = lcost_sh_curr_H;
	
    if (shIdx - 1 > 0)
		lcost_sh_prev = lcost_sh[shIdx - 1];
    else
		lcost_sh_prev = lcost_sh_curr_L;
    barrier(CLK_LOCAL_MEM_FENCE);
    
	ushort2 v_cost0_L = lcost_sh_curr_L;
	ushort2 v_cost0_H = lcost_sh_curr_H;
    ushort2 v_cost1_L = (ushort2)(lcost_sh_curr_L.y, lcost_sh_prev.x);// , 0x5432);
    ushort2 v_cost1_H = (ushort2)(lcost_sh_curr_H.y, lcost_sh_curr_L.x); // 0x5432);
    
    ushort2 v_cost2_L = (ushort2)(lcost_sh_curr_H.y, lcost_sh_curr_L.x);// 0x5432);
    ushort2 v_cost2_H = (ushort2)(lcost_sh_next.y, lcost_sh_curr_H.x);//, 0x5432);

    ushort2 v_minCost = (ushort2)(minCost, minCost);//amd_bytealign(minCost, minCost, 0x1010);
    
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
    d_scost[DISP_SIZE * idx + k * 4 + 0] += cost_tmp_L.y;
    d_scost[DISP_SIZE * idx + k * 4 + 1] += cost_tmp_L.x;
    d_scost[DISP_SIZE * idx + k * 4 + 2] += cost_tmp_H.y;
    d_scost[DISP_SIZE * idx + k * 4 + 3] += cost_tmp_H.x;
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
	int minCost = 0;

    for (int j = 0; j < width; j++) {
		minCost = stereo_loop_128(get_idx_y_0(height, i), get_idx_x_0(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

kernel void compute_stereo_horizontal_dir_kernel_4(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);
	int i = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	int minCost = 0;
//#pragma unroll
	for (int j = 0; j < width; j++) {
		minCost = stereo_loop_128(get_idx_y_4(height, i), get_idx_x_4(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

kernel void compute_stereo_vertical_dir_kernel_2(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);
	int j = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	int minCost = 0;
	//#pragma unroll
	for (int i = 0; i < height; i++) {
		minCost = stereo_loop_128(get_idx_y_2(height, i), get_idx_x_2(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


kernel void compute_stereo_vertical_dir_kernel_6(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);
	int j = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	int minCost = 0;
	//#pragma unroll
	for (int i = 0; i < height; i++) {
		minCost = stereo_loop_128(get_idx_y_6(height, i), get_idx_x_6(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}



int get_idx_x_1(int width, int j) { return j; }
int get_idx_y_1(int height, int i) { return i; }
int get_idx_x_3(int width, int j) { return width - 1 - j; }
int get_idx_y_3(int height, int i) { return i; }
int get_idx_x_5(int width, int j) { return width - 1 - j; }
int get_idx_y_5(int height, int i) { return height - 1 - i; }
int get_idx_x_7(int width, int j) { return j; }
int get_idx_y_7(int height, int i) { return height - 1 - i; }

kernel void compute_stereo_oblique_dir_kernel_1(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);
	
	const int num_paths = width + height - 1;
	int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	if (pathIdx >= num_paths) { return; }

	int i = max(0, -(width - 1) + pathIdx);
	int j = max(0, width - 1 - pathIdx);

	int minCost = 0;

	//#pragma unroll
	while (i < height && j < width) {
		minCost = stereo_loop_128(get_idx_y_1(height, i), get_idx_x_1(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
		i++; j++;
	}
}


kernel void compute_stereo_oblique_dir_kernel_3(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);

	const int num_paths = width + height - 1;
	int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	if (pathIdx >= num_paths) { return; }

	int i = max(0, -(width - 1) + pathIdx);
	int j = max(0, width - 1 - pathIdx);

	int minCost = 0;

	//#pragma unroll
	while (i < height && j < width) {
		minCost = stereo_loop_128(get_idx_y_3(height, i), get_idx_x_3(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
		i++; j++;
	}
}

kernel void compute_stereo_oblique_dir_kernel_5(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);

	const int num_paths = width + height - 1;
	int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	if (pathIdx >= num_paths) { return; }

	int i = max(0, -(width - 1) + pathIdx);
	int j = max(0, width - 1 - pathIdx);

	int minCost = 0;

	//#pragma unroll
	while (i < height && j < width) {
		minCost = stereo_loop_128(get_idx_y_5(height, i), get_idx_x_5(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
		i++; j++;
	}
}

kernel void compute_stereo_oblique_dir_kernel_7(
	global const uchar4 * d_matching_cost, global uint16_t *d_scost, int width, int height)
{
	local ushort2 lcost_sh[DISP_SIZE * PATHS_IN_BLOCK / 2];
	local ushort minCostNext[32 * PATHS_IN_BLOCK];

	init_lcost_sh_128(lcost_sh);

	const int num_paths = width + height - 1;
	int pathIdx = get_group_id(0) * PATHS_IN_BLOCK + get_local_id(1);
	if (pathIdx >= num_paths) { return; }

	int i = max(0, -(width - 1) + pathIdx);
	int j = max(0, width - 1 - pathIdx);

	int minCost = 0;

	//#pragma unroll
	while (i < height && j < width) {
		minCost = stereo_loop_128(get_idx_y_7(height, i), get_idx_x_7(width, j), d_matching_cost, d_scost, width, height, minCost, lcost_sh, minCostNext);
		//if (i == 345)
		//	printf("asdasda %d \n", minCost);
		barrier(CLK_LOCAL_MEM_FENCE);
		i++; j++;
	}
}


#define WTA_PIXEL_IN_BLOCK 8


kernel void winner_takes_all_kernel128(global ushort * leftDisp, global ushort * rightDisp, global const ushort * d_cost, int width, int height)
{
	const float uniqueness = 0.95f;

	int idx = get_local_id(0);
	int x = get_group_id(0) * WTA_PIXEL_IN_BLOCK + get_local_id(1);
	int y = get_group_id(1);

	const unsigned cost_offset = DISP_SIZE * (y * width + x);
	global const ushort* current_cost = d_cost + cost_offset;
	
	local ushort tmp_costs_block[DISP_SIZE * WTA_PIXEL_IN_BLOCK];
	local ushort * tmp_costs = tmp_costs_block + DISP_SIZE * get_local_id(1);

	uint32_t tmp_cL1, tmp_cL2; uint32_t tmp_cL3, tmp_cL4;
	uint32_t tmp_cR1, tmp_cR2; uint32_t tmp_cR3, tmp_cR4;

	// right (1)
	const int idx_1 = idx * 4 + 0;
	const int idx_2 = idx * 4 + 1;
	const int idx_3 = idx * 4 + 2;
	const int idx_4 = idx * 4 + 3;

	// TODO optimize global memory loads
	tmp_costs[idx_1] = ((x + (idx_1)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_1)) + idx_1]; // d_cost[y][x + idx0][idx0]
	tmp_costs[idx_2] = ((x + (idx_2)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_2)) + idx_2];
	tmp_costs[idx_3] = ((x + (idx_3)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_3)) + idx_3];
	tmp_costs[idx_4] = ((x + (idx_4)) >= width) ? 0xffff : d_cost[DISP_SIZE * (y * width + (x + idx_4)) + idx_4];

	//tmp_costs[idx_1] = d_cost[DISP_SIZE * (y * width + (x + idx_1)) + idx_1]; // d_cost[y][x + idx0][idx0]


	ushort4 tmp_vcL1 = vload4(0, current_cost + idx_1);
	//const uint2 idx_v = make_uint2((idx_2 << 16) | idx_1, (idx_4 << 16) | idx_3);
	//ushort4 idx_v = (ushort4)(idx_1, idx_2 , idx_3, idx_4);

	tmp_cR1 = tmp_costs[idx_1];
	tmp_cR2 = tmp_costs[idx_2];
	tmp_cR3 = tmp_costs[idx_3];
	tmp_cR4 = tmp_costs[idx_4];

	tmp_cL1 = (tmp_vcL1.x << 16) + idx_1;// __byte_perm(idx_v.x, tmp_vcL1.x, 0x5410);
	tmp_cL2 = (tmp_vcL1.y << 16) + idx_2;//__byte_perm(idx_v.x, tmp_vcL1.x, 0x7632);
	tmp_cL3 = (tmp_vcL1.z << 16) + idx_3; //__byte_perm(idx_v.y, tmp_vcL1.y, 0x5410);
	tmp_cL4 = (tmp_vcL1.w << 16) + idx_4; //__byte_perm(idx_v.y, tmp_vcL1.y, 0x7632);

	tmp_cR1 = (tmp_cR1 << 16) + idx_1;
	tmp_cR2 = (tmp_cR2 << 16) + idx_2;
	tmp_cR3 = (tmp_cR3 << 16) + idx_3;
	tmp_cR4 = (tmp_cR4 << 16) + idx_4;
	//////////////////////////////////////

	local int valL1[32 * WTA_PIXEL_IN_BLOCK];

	valL1[idx + get_local_id(1) * 32] = min(min(tmp_cL1, tmp_cL2), min(tmp_cL3, tmp_cL4));
	int minTempL1 = min_warp_int(valL1);

	int minCostL1 = (minTempL1 >> 16);
	int minDispL1 = minTempL1 & 0xffff;
	//////////////////////////////////////
	if (idx_1 == minDispL1) { tmp_cL1 = 0x7fffffff; }
	if (idx_2 == minDispL1) { tmp_cL2 = 0x7fffffff; }
	if (idx_3 == minDispL1) { tmp_cL3 = 0x7fffffff; }
	if (idx_4 == minDispL1) { tmp_cL4 = 0x7fffffff; }

	valL1[idx + get_local_id(1) * 32] = min(min(tmp_cL1, tmp_cL2), min(tmp_cL3, tmp_cL4));
	int minTempL2 = min_warp_int(valL1);
	int minCostL2 = (minTempL2 >> 16);
	int minDispL2 = minTempL2 & 0xffff;
	minDispL2 = minDispL2 == 0xffff ? -1 : minDispL2;
	//////////////////////////////////////

	if (idx_1 + x >= width) { tmp_cR1 = 0x7fffffff; }
	if (idx_2 + x >= width) { tmp_cR2 = 0x7fffffff; }
	if (idx_3 + x >= width) { tmp_cR3 = 0x7fffffff; }
	if (idx_4 + x >= width) { tmp_cR4 = 0x7fffffff; }

	valL1[idx + get_local_id(1) * 32] = min(min(tmp_cR1, tmp_cR2), min(tmp_cR3, tmp_cR4));
	int minTempR1 = min_warp_int(valL1);

	int minCostR1 = (minTempR1 >> 16);
	int minDispR1 = minTempR1 & 0xffff;
	if (minDispR1 == 0xffff) { minDispR1 = -1; }
	///////////////////////////////////////////////////////////////////////////////////
	// right (2)
	tmp_costs[idx_1] = ((idx_1) == minDispR1 || (x + (idx_1)) >= width) ? 0xffff : tmp_costs[idx_1];
	tmp_costs[idx_2] = ((idx_2) == minDispR1 || (x + (idx_2)) >= width) ? 0xffff : tmp_costs[idx_2];
	tmp_costs[idx_3] = ((idx_3) == minDispR1 || (x + (idx_3)) >= width) ? 0xffff : tmp_costs[idx_3];
	tmp_costs[idx_4] = ((idx_4) == minDispR1 || (x + (idx_4)) >= width) ? 0xffff : tmp_costs[idx_4];

	tmp_cR1 = tmp_costs[idx_1];
	tmp_cR1 = (tmp_cR1 << 16) + idx_1;

	tmp_cR2 = tmp_costs[idx_2];
	tmp_cR2 = (tmp_cR2 << 16) + idx_2;

	tmp_cR3 = tmp_costs[idx_3];
	tmp_cR3 = (tmp_cR3 << 16) + idx_3;

	tmp_cR4 = tmp_costs[idx_4];
	tmp_cR4 = (tmp_cR4 << 16) + idx_4;

	if (idx_1 + x >= width || idx_1 == minDispR1) { tmp_cR1 = 0x7fffffff; }
	if (idx_2 + x >= width || idx_2 == minDispR1) { tmp_cR2 = 0x7fffffff; }
	if (idx_3 + x >= width || idx_3 == minDispR1) { tmp_cR3 = 0x7fffffff; }
	if (idx_4 + x >= width || idx_4 == minDispR1) { tmp_cR4 = 0x7fffffff; }

	valL1[idx + get_local_id(1) * 32] = min(min(tmp_cR1, tmp_cR2), min(tmp_cR3, tmp_cR4));
	int minTempR2 = min_warp_int(valL1);
	int minCostR2 = (minTempR2 >> 16);
	int minDispR2 = minTempR2 & 0xffff;
	if (minDispR2 == 0xffff) { minDispR2 = -1; }
	///////////////////////////////////////////////////////////////////////////////////

	if (idx == 0) {
		float lhv = minCostL2 * uniqueness;
		leftDisp[y * width + x] = (lhv < minCostL1 && abs(minDispL1 - minDispL2) > 1) ? 0 : minDispL1 + 1; // add "+1" 
		float rhv = minCostR2 * uniqueness;
		rightDisp[y * width + x] = (rhv < minCostR1 && abs(minDispR1 - minDispR2) > 1) ? 0 : minDispR1 + 1; // add "+1" 
	}
}


kernel void check_consistency_kernel_left(
	global ushort* d_leftDisp, global const ushort* d_rightDisp, 
	global const uchar* d_left, int width, int height) {

	const int j = get_global_id(0);
	const int i = get_global_id(1);

	// left-right consistency check, only on leftDisp, but could be done for rightDisp too

	uchar mask = d_left[i * width + j];
	int d = d_leftDisp[i * width + j];
	int k = j - d;
	if (mask == 0 || d <= 0 || (k >= 0 && k < width && abs(d_rightDisp[i * width + k] - d) > 1)) {
		// masked or left-right inconsistent pixel -> invalid
		d_leftDisp[i * width + j] = 0;
	}
}


// clamp condition
inline int clampBC(const int x, const int y, const int nx, const int ny)
{
	const int idx = clamp(x, 0, nx - 1);
	const int idy = clamp(y, 0, ny - 1);
	return idx + idy * nx;
}

__kernel void median3x3(
	const __global ushort* restrict input,
	__global ushort* restrict output,
	const int nx,
	const int ny
)
{
	const int idx = get_global_id(0);
	const int idy = get_global_id(1);
	const int id = idx + idy * nx;

	if (idx >= nx || idy >= ny)
		return;

	ushort window[9];

	window[0] = input[clampBC(idx - 1, idy - 1, nx, ny)];
	window[1] = input[clampBC(idx, idy - 1, nx, ny)];
	window[2] = input[clampBC(idx + 1, idy - 1, nx, ny)];

	window[3] = input[clampBC(idx - 1, idy, nx, ny)];
	window[4] = input[clampBC(idx, idy, nx, ny)];
	window[5] = input[clampBC(idx + 1, idy, nx, ny)];

	window[6] = input[clampBC(idx - 1, idy + 1, nx, ny)];
	window[7] = input[clampBC(idx, idy + 1, nx, ny)];
	window[8] = input[clampBC(idx + 1, idy + 1, nx, ny)];

	// perform partial bitonic sort to find current median
	ushort flMin = min(window[0], window[1]);
	ushort flMax = max(window[0], window[1]);
	window[0] = flMin;
	window[1] = flMax;

	flMin = min(window[3], window[2]);
	flMax = max(window[3], window[2]);
	window[3] = flMin;
	window[2] = flMax;

	flMin = min(window[2], window[0]);
	flMax = max(window[2], window[0]);
	window[2] = flMin;
	window[0] = flMax;

	flMin = min(window[3], window[1]);
	flMax = max(window[3], window[1]);
	window[3] = flMin;
	window[1] = flMax;

	flMin = min(window[1], window[0]);
	flMax = max(window[1], window[0]);
	window[1] = flMin;
	window[0] = flMax;

	flMin = min(window[3], window[2]);
	flMax = max(window[3], window[2]);
	window[3] = flMin;
	window[2] = flMax;

	flMin = min(window[5], window[4]);
	flMax = max(window[5], window[4]);
	window[5] = flMin;
	window[4] = flMax;

	flMin = min(window[7], window[8]);
	flMax = max(window[7], window[8]);
	window[7] = flMin;
	window[8] = flMax;

	flMin = min(window[6], window[8]);
	flMax = max(window[6], window[8]);
	window[6] = flMin;
	window[8] = flMax;

	flMin = min(window[6], window[7]);
	flMax = max(window[6], window[7]);
	window[6] = flMin;
	window[7] = flMax;

	flMin = min(window[4], window[8]);
	flMax = max(window[4], window[8]);
	window[4] = flMin;
	window[8] = flMax;

	flMin = min(window[4], window[6]);
	flMax = max(window[4], window[6]);
	window[4] = flMin;
	window[6] = flMax;

	flMin = min(window[5], window[7]);
	flMax = max(window[5], window[7]);
	window[5] = flMin;
	window[7] = flMax;

	flMin = min(window[4], window[5]);
	flMax = max(window[4], window[5]);
	window[4] = flMin;
	window[5] = flMax;

	flMin = min(window[6], window[7]);
	flMax = max(window[6], window[7]);
	window[6] = flMin;
	window[7] = flMax;

	flMin = min(window[0], window[8]);
	flMax = max(window[0], window[8]);
	window[0] = flMin;
	window[8] = flMax;

	window[4] = max(window[0], window[4]);
	window[5] = max(window[1], window[5]);

	window[6] = max(window[2], window[6]);
	window[7] = max(window[3], window[7]);

	window[4] = min(window[4], window[6]);
	window[5] = min(window[5], window[7]);

	output[id] = min(window[4], window[5]);
}



kernel void copy_u8_to_u16(global const uchar * input,
	global ushort * output)
{
	int x = get_global_id(0);
	output[x] = input[x];
}

kernel void clear_buffer(global float8 * buff)
{
	int x = get_global_id(0);
	buff[x] = (float8)0;
}