cmake_minimum_required(VERSION 3.0)

project(sgm-cl)


find_package(OpenCL REQUIRED)

add_library(sgm-cl sgm-cl.cpp sgm-cl/sgm-cl.h sgm-cl/sgm-cl.cl)
add_library(::sgm-cl ALIAS sgm-cl)
set_target_properties(sgm-cl PROPERTIES DEBUG_POSTFIX _d)
message(STATUS "OpenCL libraries ${OpenCL_LIBRARIES}")
target_link_libraries(sgm-cl PUBLIC ${OpenCL_LIBRARIES})
target_include_directories(sgm-cl 
    PUBLIC 
        ${OpenCL_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/sgm-cl>
    )

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gen/${PROJECT_NAME}-config-version.cmake"
  VERSION 0.0.1
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
   sgm-cl/sgm-cl.h 
   sgm-cl/sgm-cl.cl
   DESTINATION include/sgm-cl/sgm-cl)
