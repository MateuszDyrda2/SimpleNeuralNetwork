cmake_minimum_required(VERSION 3.21)

set(biai_VERSION_MAJOR 0)
set(biai_VERSION_MINOR 1)
set(biai_VERSION_PATCH 0)
set(biai_VERSION ${biai_VERSION_MAJOR}.${biai_VERSION_MINOR}.${biai_VERSION_PATCH})

# libpng
# find_package(ZLIB REQUIRED)
# find_package(PNG REQUIRED)
add_subdirectory(external/zlib)
add_subdirectory(external/libpng)
# eigen
add_subdirectory(external/eigen)

project(biai VERSION ${biai_VERSION})

set(CMAKE_CXX_VERSION 20)
set(CMAKE_CXX_VERSION_REQUIRED ON)


set(biai_BUILD_EXAMPLE ON)

# the library is header only
add_library(biai INTERFACE)
target_include_directories(biai INTERFACE include/)
target_link_libraries(biai INTERFACE Eigen3::Eigen)
set_property(TARGET biai PROPERTY CXX_STANDARD 20)


if(${biai_BUILD_EXAMPLE})
	add_executable(biai_example example/main.cpp)
	target_link_libraries(biai_example PUBLIC biai png)
	target_include_directories(biai_example PUBLIC ${PNG_INCLUDE_DIRS})
	set_property(TARGET biai_example PROPERTY CXX_STANDARD 20)
endif()