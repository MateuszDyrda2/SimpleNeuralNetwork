cmake_minimum_required(VERSION 3.21)

project(biai)

set(biai_SRC
	src/network.cpp
)
set(biai_BUILD_EXAMPLE ON)

add_library(biai ${biai_SRC})
target_include_directories(biai PUBLIC include/)

if(${biai_BUILD_EXAMPLE})
	add_executable(example example/main.cpp)
	target_link_libraries(example PUBLIC biai)
endif()