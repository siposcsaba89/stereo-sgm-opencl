
inline uint32_t pack_uint8x4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) 
{
    uchar4 uint8x4;
    uint8x4.x = (uint8_t)(x);
    uint8x4.y = (uint8_t)(y);
    uint8x4.z = (uint8_t)(z);
    uint8x4.w = (uint8_t)(w);
    return as_uint(uint8x4);
}

void store_uint8_vector_8u(global uint8_t* dest, const uint32_t* ptr)
{
    uint2 uint32x2;
    uint32x2.x = pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]);
    uint32x2.y = pack_uint8x4(ptr[4], ptr[5], ptr[6], ptr[7]);
    global uint2* dest_ptr = (global uint2*) dest;
    *dest_ptr = uint32x2;
}

void store_uint8_vector(global uint8_t* dest, const int N,
    const uint32_t* ptr)
{
    if (N == 16)
    {
        uchar16 vv;
        uchar* v = (uchar*)&vv;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            v[i] = (uint8_t)ptr[i];
        *((global uchar16*)dest) = vv;
    }
    else if (N == 8)
    {
        uchar8 vv;
        uchar* v = (uchar*)&vv;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            v[i] = (uint8_t)ptr[i];
        *((global uchar8*)dest) = vv;

    }
    else
    {
#pragma unroll
        for (int i = 0; i < N; ++i)
            dest[i] = (uint8_t)ptr[i];
    }
}


inline void load_uint8_vector(uint32_t* dest, const int num, const local uint8_t* ptr) 
{
#pragma unroll
    for (int  i = 0; i < num; ++i)
        dest[i] = (uint32_t)(ptr[i]);
    //barrier(CLK_LOCAL_MEM_FENCE);
}


inline void g_load_uint8_vector(uint32_t* dest, const int num, const global uint8_t* ptr)
{
    if (num == 16)
    {
        uchar16 vv = *((global uchar16*)ptr);
        uchar* v = (uchar*)&vv;
#pragma unroll
        for (int i = 0; i < 16; ++i)
            dest[i] = (uint32_t)v[i];   
    }
    else if (num == 8)
    {
        uchar8 vv = *((global uchar8*)ptr);
        uchar* v = (uchar*)&vv;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            dest[i] = (uint32_t)v[i];
    }
    else
    {
#pragma unroll
        for (int i = 0; i < num; ++i)
            dest[i] = (uint32_t)(ptr[i]);
    }
}


inline void lload_uint8_vector(uint32_t* dest, const int num,  const uint8_t* ptr)
{
    for (int i = 0; i < num; ++i)
        dest[i] = (uint32_t)(ptr[i]);
}

inline void load_uint16_vector(uint32_t* dest, const int num, const local uint16_t* ptr)
{
    for (int i = 0; i < num; ++i)
        dest[i] = (uint32_t)(ptr[i]);

}


inline void lload_uint16_vector(uint32_t* dest, const int num, const uint16_t* ptr)
{
    for (int i = 0; i < num; ++i)
        dest[i] = (uint32_t)(ptr[i]);
}


inline void store_uint16_vector(local uint16_t* dest, const int N, const uint32_t* ptr)
{
    for (int i = 0; i < N; ++i)
        dest[i] = (uint16_t)ptr[i];
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline uint32_t subgroup_min(const uint32_t lane_id,
    const uint32_t subgroup_size,
    local uint32_t* shfl_mem)
{
    int lid = get_local_id(0);
    for (int i = subgroup_size / 2; i > 0; i >>= 1)
    {
        if (lane_id < i)
        {
            shfl_mem[lid] = min(shfl_mem[lid], shfl_mem[lid + i]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int sub_group_idx = get_local_id(0) / subgroup_size;
    return shfl_mem[sub_group_idx * subgroup_size];
}


