
#define INVALID_DISP ((uint16_t)(-1))
kernel void correct_disparity_range_kernel(global uint16_t* d_disp,
    int width,
    int height,
    int pitch,
    int min_disp_scaled,
    int invalid_disp_scaled)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
    {
        return;
    }

    uint16_t d = d_disp[y * pitch + x];
    if (d == INVALID_DISP)
    {
        d = invalid_disp_scaled;
    }
    else
    {
        d += min_disp_scaled;
    }
    d_disp[y * pitch + x] = d;
}

kernel void cast_16bit_8bit_array_kernel( const global uint16_t* arr16bits,
    global uint8_t* arr8bits,
    int num_elements)
{
    int i = get_global_id(0);
    if (i >= num_elements)
        return;
    arr8bits[i] = (uint8_t)arr16bits[i];
}
