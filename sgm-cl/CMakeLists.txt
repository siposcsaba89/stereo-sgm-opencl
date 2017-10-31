cmake_minimum_required(VERSION 3.0)

project(sgm-cl)


find_package(napalm REQUIRED)

add_library(sgm-cl sgm-cl.cpp sgm-cl/sgm-cl.h)
add_library(::sgm-cl ALIAS sgm-cl)

set(PROGRAMS sgm-cl/sgm.cl)
set(EMBED_PROGRAM_FILES TRUE)
set(COMPILE_IN_BUILD_TYPE_IF_POSSIBLE TRUE)
set(PROGRAM_RUNTIME_SOURCE_DIR "")
create_programs("${PROGRAMS}"
        output 
        sgm-cl 
        ${COMPILE_IN_BUILD_TYPE_IF_POSSIBLE} 
        ${EMBED_PROGRAM_FILES}
        ${PROGRAM_RUNTIME_SOURCE_DIR}
)


set_target_properties(sgm-cl PROPERTIES DEBUG_POSTFIX _d)
target_include_directories(sgm-cl 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/sgm-cl>
    )

target_link_libraries(sgm-cl PUBLIC ::napalm)
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
   sgm-cl/sgm.cl
   DESTINATION include/sgm-cl/sgm-cl)
