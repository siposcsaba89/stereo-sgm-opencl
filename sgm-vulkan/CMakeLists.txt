cmake_minimum_required(VERSION 3.0)

project(sgm-vulkan)


find_package(Vulkan REQUIRED)

add_library(sgm-vulkan sgm-vulkan.cpp sgm-vulkan.h)
add_library(::sgm-vulkan ALIAS sgm-vulkan)
target_link_libraries(sgm-vulkan PUBLIC ${VULKAN_LIBRARY})
target_include_directories(sgm-vulkan PUBLIC ${VULKAN_INCLUDE_DIR})
