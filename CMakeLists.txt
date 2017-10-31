cmake_minimum_required(VERSION 3.0)


project(stereo-sgm)

option(BUILD_SGM_CL "Build sgm using opnecl!" ON)
option(BUILD_SGM_VULKAN "Build sgm using VULKAN!" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


find_package(OpenCV REQUIRED)
find_package(napalm REQUIRED)

if (BUILD_SGM_CL)
    add_subdirectory(sgm-cl)
    add_executable(sgm-cl-test sgm-cl-test.cpp)
    target_link_libraries(sgm-cl-test PRIVATE ::sgm-cl ${OpenCV_LIBRARIES})
    target_include_directories(sgm-cl-test PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if (BUILD_SGM_VULKAN)
    add_subdirectory(sgm-vulkan)
    add_executable(sgm-vulkan-test sgm-vulkan-test.cpp)
    target_link_libraries(sgm-vulkan-test PRIVATE ::sgm-vulkan ${OpenCV_LIBRARIES})
    target_include_directories(sgm-vulkan-test PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
