cmake_minimum_required(VERSION 3.21)

project("Boundary Layer" VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(devel_profile
    src/exec/develop_flat_plate.cpp
    src/profile.cpp
)
target_include_directories(devel_profile PRIVATE src)

add_executable(search_profile
    src/exec/search_flat_plate.cpp
    src/profile_search.cpp
    src/profile.cpp
)
target_include_directories(search_profile PRIVATE src)
