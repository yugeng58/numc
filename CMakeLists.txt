cmake_minimum_required(VERSION 3.26)
project(ndarray LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)  # 必须存在REQUIRED

add_executable(ndarray
        main.cpp
        numc.cpp
        numc_modify.cpp
        numc_safe.cpp
        ndarray.cpp
        ndarray_parallel.cpp
        ndarray_safe.cpp
)

# 强制应用OpenMP设置到目标
target_compile_options(ndarray PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(ndarray PRIVATE OpenMP::OpenMP_CXX)

# 添加以下调试信息
message(STATUS "OpenMP CXX flags: " ${OpenMP_CXX_FLAGS})
message(STATUS "OpenMP libs: " ${OpenMP_CXX_LIBRARIES})