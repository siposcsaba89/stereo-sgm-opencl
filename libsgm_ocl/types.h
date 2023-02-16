/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef SGM_TYPES_HPP
#define SGM_TYPES_HPP

#include <cstdint>

namespace sgm
{

namespace cl
{
/**
 Indicates number of scanlines which will be used.
*/
enum class PathType : uint8_t
{
    SCAN_4PATH = 4, //>! Horizontal and vertical paths.
    SCAN_8PATH = 8  //>! Horizontal, vertical and oblique paths.
};
enum class MaxDisparity : uint16_t
{
    MAX_DISP_64 = 64,
    MAX_DISP_128 = 128,
    MAX_DISP_256 = 256
};

enum class DispPrecision : uint8_t
{
    INTEGER = 0,
    SUBPIXEL = 1
};

} // namespace cl
} // namespace sgm

#endif