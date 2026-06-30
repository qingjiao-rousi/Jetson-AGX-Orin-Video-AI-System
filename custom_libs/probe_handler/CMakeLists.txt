cmake_minimum_required(VERSION 3.16)

project(probe_handler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(probe_handler SHARED
    src/probe_meta_parser.cpp
)

target_include_directories(probe_handler
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(probe_handler PRIVATE -Wall -Wextra -Wpedantic)

find_path(DEEPSTREAM_INCLUDE_DIR
    NAMES nvdsmeta.h
    PATHS
        /opt/nvidia/deepstream/deepstream/sources/includes
        /opt/nvidia/deepstream/deepstream-${NVDS_VERSION_MAJOR}/sources/includes
)

if(DEEPSTREAM_INCLUDE_DIR)
    target_include_directories(probe_handler PRIVATE ${DEEPSTREAM_INCLUDE_DIR})
else()
    message(WARNING "DeepStream headers not found; probe_handler will build with the offline stub path.")
endif()
