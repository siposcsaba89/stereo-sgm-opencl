cmake_minimum_required(VERSION 3.0)

project(sgm-cl)


find_package(OpenCL REQUIRED)

add_library(sgm-cl sgm-cl.cpp sgm-cl.h sgm-cl.cl)
add_library(::sgm-cl ALIAS sgm-cl)
target_link_libraries(sgm-cl PUBLIC ${OpenCL_LIBRARIES})
target_include_directories(sgm-cl PUBLIC ${OpenCL_INCLUDE_DIRS})
