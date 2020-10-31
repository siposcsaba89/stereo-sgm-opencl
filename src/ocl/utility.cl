
inline uint32_t pack_uint8x4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) 
{
    uchar4 uint8x4;
    uint8x4.x = (uint8_t)(x);
    uint8x4.y = (uint8_t)(y);
    uint8x4.z = (uint8_t)(z);
    uint8x4.w = (uint8_t)(w);
    return as_uint(uint8x4);
}

void store_uint8_vector_16u(global uint8_t* dest,
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
