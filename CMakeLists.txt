cmake_minimum_required(VERSION 3.21)

project("Boundary Layer" VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

###
# Define libraries
#

add_library(utils_lib
    src/utils/utils.cpp
)
target_include_directories(utils_lib PUBLIC src/utils)


add_library(profile_lib
    src/profile.cpp
)
target_include_directories(profile_lib PRIVATE src src/physics)
target_link_libraries(profile_lib PRIVATE utils_lib)

add_library(flat_plate_lib
    src/flat_plate.cpp
    src/flat_plate_factory.cpp
)
target_include_directories(flat_plate_lib PRIVATE src src/physics)
target_link_libraries(flat_plate_lib PRIVATE utils_lib)
target_link_libraries(flat_plate_lib PRIVATE profile_lib)

###
# Define executables
#

add_executable(devel_profile
    src/exec/develop_flat_plate.cpp
)

target_link_libraries(devel_profile PRIVATE flat_plate_lib profile_lib utils_lib)
target_include_directories(devel_profile PRIVATE src)

add_executable(search_profile
    src/exec/search_flat_plate.cpp
)

target_link_libraries(search_profile PRIVATE flat_plate_lib profile_lib utils_lib)
target_include_directories(search_profile PRIVATE src src/physics)

###
# Tests
#
enable_testing()

add_subdirectory(tests)