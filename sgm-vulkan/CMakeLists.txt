cmake_minimum_required(VERSION 3.0)

project(sgm-vulkan)


find_package(Vulkan REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(sgm-vulkan 
    #sgm-vulkan.cpp sgm-vulkan.h
    computeshader.cpp
    vulkanbuffer.hpp
    vulkanexamplebase.cpp
    vulkanexamplebase.h
    vulkantools.h
    vulkantools.cpp
    VulkanInitializers.hpp
    vulkandevice.hpp
    vulkandebug.h
    vulkandebug.cpp
    vulkanTextureLoader.hpp
    vulkanswapchain.hpp
    vulkantextoverlay.hpp
    missing.hpp
    )
#add_library(::sgm-vulkan ALIAS sgm-vulkan)
target_link_libraries(sgm-vulkan PUBLIC ${VULKAN_LIBRARY} Vulkan::Vulkan ${OpenCV_LIBRARIES})
target_include_directories(sgm-vulkan PUBLIC ${VULKAN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(sgm-vulkan PUBLIC -DVK_USE_PLATFORM_WIN32_KHR)
