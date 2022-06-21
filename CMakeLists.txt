cmake_minimum_required(VERSION 3.21)

set(biai_VERSION_MAJOR 1)
set(biai_VERSION_MINOR 0)
set(biai_VERSION_PATCH 0)
set(biai_VERSION ${biai_VERSION_MAJOR}.${biai_VERSION_MINOR}.${biai_VERSION_PATCH})

# libpng
# find_package(PNG REQUIRED)
#add_subdirectory(external/zlib)
set(PNG_TESTS OFF CACHE BOOL "" FORCE)
set(PNG_SHARED ON CACHE BOOL "" FORCE)

add_subdirectory(external/libpng)

project(biai VERSION ${biai_VERSION})

set(biai_BUILD_EXAMPLE ON)

# the library is header only
add_library(biai INTERFACE)
target_include_directories(biai INTERFACE include/)
set_property(TARGET biai PROPERTY CXX_STANDARD 20)

if(${biai_BUILD_EXAMPLE})
	add_executable(biai_example example/main.cpp)
	target_link_libraries(biai_example PUBLIC biai png)
	target_include_directories(biai_example PUBLIC ${PNG_INCLUDE_DIRS})
	set_property(TARGET biai_example PROPERTY CXX_STANDARD 20)
endif()