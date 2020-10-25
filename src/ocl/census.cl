// pixel and feature type defs for example
//#define pixel_type uint8_t
@pixel_type@
// numeric types
#define uint32_t uint
#define uint16_t ushort
#define uint8_t uchar

#define int64_t long
#define int32_t int
#define int16_t short
#define int8_t char
// end numeric type defs


#define feature_type uint32_t

// census transfrom defines
#define WINDOW_WIDTH  9
#define WINDOW_HEIGHT  7
#define BLOCK_SIZE_CENSUS 128
#define LINES_PER_BLOCK 16
#define SMEM_BUFFER_SIZE WINDOW_HEIGHT + 1


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
