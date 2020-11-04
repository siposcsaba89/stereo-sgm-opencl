
#define INVALID_DISP ((uint16_t)(-1))
#define DST_T uint16_t
@SRC_T@
@SUBPIXEL_SHIFT@

kernel void check_consistency_kernel(global DST_T* d_leftDisp,
    global const DST_T* d_rightDisp,
    global const SRC_T* d_left,
    int width,
    int height,
    int src_pitch,
    int dst_pitch,
    int subpixel, 
    int LR_max_diff) 
{

    const int j = get_global_id(0);
    const int i = get_global_id(1);
    if (i >= height || j >= width)
        return;

    // left-right consistency check, only on leftDisp, but could be done for rightDisp too

    SRC_T mask = d_left[i * src_pitch + j];
    DST_T org = d_leftDisp[i * dst_pitch + j];
    int d = org;
    if (subpixel == 1) 
    {
        d >>= SUBPIXEL_SHIFT;
    }
    int k = j - d;
    if (mask == 0 ||
        org == INVALID_DISP ||
        (k >= 0 && k < width && LR_max_diff >= 0 && abs(d_rightDisp[i * dst_pitch + k] - d) > LR_max_diff))
    {
        // masked or left-right inconsistent pixel -> invalid
        d_leftDisp[i * dst_pitch + j] = (DST_T)(INVALID_DISP);
    }
}