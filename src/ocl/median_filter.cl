
// clamp condition
inline int clampBC(const int x, const int y, const int nx, const int ny, const int pitch)
{
    const int idx = clamp(x, 0, nx - 1);
    const int idy = clamp(y, 0, ny - 1);
    return idx + idy * pitch;
}

kernel void median3x3(
    const global ushort* restrict input,
    global ushort* restrict output,
    const int nx,
    const int ny,
    const int pitch
)
{
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);
    const int id = idx + idy * pitch;

    if (idx >= nx || idy >= ny)
        return;

    ushort window[9];

    window[0] = input[clampBC(idx - 1, idy - 1, nx, ny, pitch)];
    window[1] = input[clampBC(idx, idy - 1, nx, ny, pitch)];
    window[2] = input[clampBC(idx + 1, idy - 1, nx, ny, pitch)];

    window[3] = input[clampBC(idx - 1, idy, nx, ny, pitch)];
    window[4] = input[clampBC(idx, idy, nx, ny, pitch)];
    window[5] = input[clampBC(idx + 1, idy, nx, ny, pitch)];

    window[6] = input[clampBC(idx - 1, idy + 1, nx, ny, pitch)];
    window[7] = input[clampBC(idx, idy + 1, nx, ny, pitch)];
    window[8] = input[clampBC(idx + 1, idy + 1, nx, ny, pitch)];

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
