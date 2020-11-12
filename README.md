# libSGM_ocl
# SGM implementation in OpenCL V2.0

Computes dense stereo correspondences.

Max disparities could be one of 64, 128, 256.

 Using lower max disparity number increases performance and decreases memory usage.


4 path optimization optimizes disparity cost from `left - > right, right -> left, up -> down, down-> up`

8 path optimization adds oblique directions:
` upleft -> downright, upright -> downleft, downright -> upleft, downleft -> upright`

If subpixel disparity is enabled, the result is multiplied by 16. You can calculate the floating point disparities by dividing the result by 16: `float d = res / 16.0f;`

## Dependencies
- OpenCL

Install dependencies for ubuntu: `sudo apt install ocl-icd-opencl-dev`

On windows you can install OpenCL via vcpkg: `vcpkg install opencl:x64-windows`

Additional dependencies to build examples
- OpenCV
 
 Ubuntu: `sudo apt install libopencv-dev`

Vcpkg: `vcpkg install opencv:x64-windows`

## Building:

CMake options:

- BUILD_EXAMPLES - build examples, default value is ON
- CL_TARGET_OPENCL_VERSION - defines OpenCL target version, default value us 120

``` 
cmake .. -DBUILD_EXAMPLES=ON -DCL_TARGET_OPENCL_VERSION=120 -DCMAKE_BUILD_TYPE=Release
```

Note: if you are using vcpkg, please set either `CMAKE_PREFIX_PATH` or `CMAKE_TOOLCHAIN_FILE` as noted in vcpkg documentation.

```
cmake .. -DBUILD_EXAMPLES=ON -DCL_TARGET_OPENCL_VERSION=120 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake
```

## Running stereo_movie example

    Usage: stereo_movie.exe [params] img_source_left img_source_right

        --device_idx (value:0)
                OpenCL device index
        -h, --help (value:true)
                Print this message
        --max_disparity, --md (value:128)
                Maximum disparity
        --np, --num_path (value:4)
                Num path to optimize, 4 or 8
        --platform_idx (value:0)
                OpenCL plarform index
        --sp, --subpixel (value:false)
                Compute subpixel accuracy

        img_source_left (value:d:\datasets\kitti\sequences\00\image_0\000000.png)
                Left images
        img_source_right (value:d:\datasets\kitti\sequences\00\image_1\000000.png)
                Right images

Run on kitti rectified stereo database:

On Windows:

```
./stereo_movie [path to kitti]\sequences\00\image_0\000000.png [path to kitti]\00\image_1\000000.png 
```

On Linux:

```
./stereo_movie [path to kitti]\sequences\00\image_0\%06d.png [path to kitti]\00\image_1\%06d.png 
```

## Known issues
 - oblique path optimizations are not working correctly, there is some magic error. 4 path optimization gives better result now, 2 x faster and uses ~ half memory of 8 path optimization

## Testing and performance

Kitti dataset, resolution **1241x376**. Parameters:
 - --subpixel=true
 - --max_disparity=128
 - --num_path=4

Nvidia Driver version: 455.32.00
AMD gpu driver, revision 20.40

Device | Windows 10 20H2 | Ubuntu 2004 
----|---------|------------ 
AMD Radeon RX 480 8GB | 12 milliseconds | 12
NVidia GTX 1070 8GB | 8 milliseconds | 7 milliseconds


<br/>

# Refs
Implementation based on libSGM:
 - https://github.com/fixstars/libSGM

vcpkg: 
- https://github.com/microsoft/vcpkg

Uses a modified version of CMakeRC:
- https://github.com/vector-of-bool/cmrc
 
Kitti:
- http://www.cvlibs.net/datasets/kitti/
