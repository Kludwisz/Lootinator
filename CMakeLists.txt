cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Lootinator9000 LANGUAGES CXX)
include(CTest)

if(MSVC)
	add_compile_options(/W4)
else()
	add_compile_options(-Wall -Wextra -pedantic)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(lootinator)
add_subdirectory(example)
