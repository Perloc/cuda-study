cmake_minimum_required(VERSION 3.22)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
    message("Setting default build type to Debug")
endif()

project(CUDA-DEV CUDA)

# 设置生成compile_commands.json文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置头文件路径
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(./src)
