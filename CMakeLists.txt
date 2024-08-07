cmake_minimum_required(VERSION 3.21)

project("Boundary Layer" VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to enable debugging
option(ENABLE_DEBUG "Enable debugging information" OFF)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(ENABLE_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  message(STATUS "Debugging enabled.\n")
endif()

###
# Define libraries
#

# Helper functions
add_library(utils_lib
    src/utils/utils.cpp
)
target_include_directories(utils_lib PUBLIC src/utils)

# Profile functions
add_library(profile_lib
    src/profile_functions_default.cpp
    src/profile_functions_cpg.cpp
)
target_include_directories(profile_lib PRIVATE src src/physics)
target_link_libraries(profile_lib PRIVATE utils_lib)

# Linear algebra functions
add_library(linalg_lib
    src/linalg/dense_matrix.cpp
    src/linalg/dense_direct_solver.cpp
)
target_include_directories(linalg_lib PRIVATE src/linalg)

# Flat Plate functions
add_library(boundary_layer_lib
    src/boundary_layer.cpp
    src/boundary_layer_factory.cpp
)
target_include_directories(boundary_layer_lib PRIVATE src src/physics src/solvers src/linalg)
target_link_libraries(boundary_layer_lib PRIVATE utils_lib)
target_link_libraries(boundary_layer_lib PRIVATE profile_lib)

###
# Define executables
#

# Develop a self-similar profile for set initial conditions
add_executable(devel_profile
    src/exec/develop_flat_plate.cpp
)

target_link_libraries(devel_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(devel_profile PRIVATE src src/physics)

# Search for the initial conditions that develop into the correct self-similar profile
add_executable(search_profile
    src/exec/search_flat_plate.cpp
)

target_link_libraries(search_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(search_profile PRIVATE src src/physics src/utils)

# Compute 2D profile
add_executable(compute_profiles
    src/exec/compute_flat_plate_local_sim.cpp
)

target_link_libraries(compute_profiles PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(compute_profiles PRIVATE src src/physics src/utils)

###
# Tests
#
enable_testing()

add_subdirectory(tests)