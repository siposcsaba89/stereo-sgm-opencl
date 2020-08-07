cmake_minimum_required(VERSION 3.10)

project(libsgm_ocl)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

option(BUILD_EXAMPLES "Build sgm ocl example" ON)

find_package(OpenCL REQUIRED)

add_library(libsgm_ocl src/libsgm_ocl.cpp libsgm_ocl/libsgm_ocl.h)
add_library(::libsgm_ocl ALIAS libsgm_ocl)

set(PROGRAMS libsgm_ocl/sgm.cl)

set_target_properties(libsgm_ocl PROPERTIES DEBUG_POSTFIX _d RELWITHDEBINFO_POSTFIX _rd)
target_include_directories(libsgm_ocl
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/sgm-cl>
)
target_link_libraries(libsgm_ocl PUBLIC OpenCL::OpenCL)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake"
  VERSION 0.0.2
  COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets DESTINATION 
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake 
        ${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake
    DESTINATION 
        lib/cmake/${PROJECT_NAME})
    
install(EXPORT ${PROJECT_NAME}-targets NAMESPACE :: DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES
   libsgm_ocl/libsgm_ocl.h 
   libsgm_ocl/sgm.cl
   DESTINATION include/libsgm_ocl/libsgm_ocl)


if (BUILD_EXAMPLES)
    find_package(OpenCV REQUIRED)
    add_executable(sgm-ocl-test sgm_ocl_test.cpp)
    target_link_libraries(sgm-ocl-test PRIVATE ::libsgm_ocl ${OpenCV_LIBRARIES})
endif()
