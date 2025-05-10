cmake_minimum_required(VERSION 3.26)
project(ndarray)

set(CMAKE_CXX_STANDARD 17)

add_executable(ndarray main.cpp
        numc.cpp
)
