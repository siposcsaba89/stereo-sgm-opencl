cmake_minimum_required(VERSION 3.0)

project(sgm-opencl)


find_package(OpenCL REQUIRED)

add_library(sgm-opencl stereo-sgm.cpp stereo-sgm.h)
target_link_libraries(sgm-opencl PUBLIC ${OpenCL_LIBRARIES})
target_include_directories(sgm-opencl PUBLIC ${OpenCL_INCLUDE_DIRS})

add_executable(sgm-opencl-test sgm-test.cpp)
target_link_libraries(sgm-opencl-test sgm-opencl)