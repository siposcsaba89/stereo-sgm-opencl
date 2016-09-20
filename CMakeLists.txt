cmake_minimum_required(VERSION 3.0)

project(sgm-opencl)


find_package(OpenCL REQUIRED)

add_library(sgm-opencl stereo-sgm.cpp stereo-sgm.h sgm-program.cl)
add_library(::sgm-opencl ALIAS sgm-opencl)
target_link_libraries(sgm-opencl PUBLIC ${OpenCL_LIBRARIES})
target_include_directories(sgm-opencl PUBLIC ${OpenCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

add_executable(sgm-opencl-test sgm-test.cpp)
target_link_libraries(sgm-opencl-test PRIVATE ::sgm-opencl ${OpenCV_LIBRARIES})
target_include_directories(sgm-opencl-test PRIVATE ${OpenCV_INCLUDE_DIRS})

#export(targets sgm-opencl)