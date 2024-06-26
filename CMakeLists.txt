cmake_minimum_required(VERSION 3.0.0)

project(tool)
 
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(tool main.cpp)
target_link_libraries(tool yaml-cpp)
message(STAUTS "Include directory: ${CMAKE_SOURCE_DIR}/include")
message(STATUS "Library directory: ${CMAKE_SOURCE_DIR}/lib")