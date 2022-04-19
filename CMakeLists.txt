cmake_minimum_required(VERSION 3.21)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_VERSION 17)
set(CMAKE_CXX_VERSION_REQUIRED ON)

set(biai_VERSION_MAJOR 0)
set(biai_VERSION_MINOR 1)
set(biai_VERSION_PATCH 0)
set(biai_VERSION ${biai_VERSION_MAJOR}.${biai_VERSION_MINOR}.${biai_VERSION_PATCH})

# libpng
set(PNG_DIRECTORY "C:/libpng/" CACHE STRING "Where to find libpng" FORCE)
find_package(PNG)
if(NOT ${PNG_FOUND})
	if(WIN32)
        set(PNG_PNG_INCLUDE_DIR ${PNG_DIRECTORY}include)
        set(PNG_LIBRARY ${PNG_DIRECTORY}lib)
		find_package(PNG REQUIRED)
	else()
		message(FATAL_ERROR "Make sure the pnglib is installed")
	endif()
endif()
include_directories(${PNG_PNG_INCLUDE_DIR})

# eigen
add_subdirectory(external/eigen)
project(biai VERSION ${biai_VERSION})

set(biai_BUILD_EXAMPLE ON)

# the library is header only
add_library(biai INTERFACE)
target_include_directories(biai INTERFACE include/)
target_link_libraries(biai INTERFACE Eigen3::Eigen)

if(${biai_BUILD_EXAMPLE})
	add_executable(example example/main.cpp)
	target_link_libraries(example PUBLIC biai libpng)
endif()
