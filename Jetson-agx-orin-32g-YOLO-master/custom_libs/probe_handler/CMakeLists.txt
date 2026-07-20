cmake_minimum_required(VERSION 3.16)

project(probe_handler LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB2 REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)

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
target_link_libraries(probe_handler PRIVATE PkgConfig::GLIB2 PkgConfig::GSTREAMER)

find_path(DEEPSTREAM_INCLUDE_DIR
    NAMES nvdsmeta.h
    PATHS
        /opt/nvidia/deepstream/deepstream/sources/includes
        /opt/nvidia/deepstream/deepstream-${NVDS_VERSION_MAJOR}/sources/includes
)

if(DEEPSTREAM_INCLUDE_DIR)
    target_include_directories(probe_handler PRIVATE ${DEEPSTREAM_INCLUDE_DIR})
    find_library(NVDS_META_LIBRARY
        NAMES nvdsgst_meta
        PATHS
            /opt/nvidia/deepstream/deepstream/lib
            /opt/nvidia/deepstream/deepstream-${NVDS_VERSION_MAJOR}/lib
        NO_DEFAULT_PATH
    )
    if(NVDS_META_LIBRARY)
        target_link_libraries(probe_handler PRIVATE ${NVDS_META_LIBRARY})
    else()
        message(WARNING "DeepStream metadata library not found; GST buffer parser may fail to link.")
    endif()
else()
    message(WARNING "DeepStream headers not found; probe_handler will build with the offline stub path.")
endif()
