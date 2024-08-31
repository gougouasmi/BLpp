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
# 3rd party libraries
#

#find_package(HDF5 REQUIRED COMPONENTS CXX)

###
# Define libraries
#

# Helper functions
add_library(utils_lib
    src/utils/utils.cpp
    src/utils/file_io.cpp
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
    src/linalg/dense_linalg.cpp
    src/linalg/dense_direct_solver.cpp
)
target_include_directories(linalg_lib PRIVATE src/linalg)

# Flat Plate functions
add_library(boundary_layer_lib
    src/boundary_layer.cpp
    src/boundary_layer_factory.cpp
)
target_include_directories(boundary_layer_lib PRIVATE src src/physics src/solvers src/linalg src/data)
target_link_libraries(boundary_layer_lib PRIVATE utils_lib)
target_link_libraries(boundary_layer_lib PRIVATE profile_lib)

# Case setup functions
add_library(cases_lib src/cases/case_functions.cpp)
target_include_directories(cases_lib PRIVATE src/ src/physics src/solvers src/linalg src/data)
target_link_libraries(cases_lib PRIVATE utils_lib)

###
# Define executables
#

# Develop a self-similar profile for set initial conditions
add_executable(devel_profile
    src/exec/develop_profile.cpp
)

target_link_libraries(devel_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(devel_profile PRIVATE src src/physics src/data)

# Search for the initial conditions that develop into the correct self-similar profile
add_executable(search_profile
    src/exec/search_profile.cpp
)

target_link_libraries(search_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(search_profile PRIVATE src src/physics src/utils src/data)

# Compute 2D profile (local-similarity)
add_executable(compute_2d_profile
    src/exec/compute_2d_profile.cpp
)

target_link_libraries(compute_2d_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib cases_lib)
target_include_directories(compute_2d_profile PRIVATE src src/physics src/utils src/cases src/data)

# Compute edge flow from pressure
add_executable(edge_solve src/exec/solve_from_pressure.cpp src/solvers/edge_solvers.cpp)

target_link_libraries(edge_solve PRIVATE utils_lib linalg_lib)
target_include_directories(edge_solve PRIVATE src src/utils src/solvers src/data src/linalg)

# (DEBUGGING) Develop profile for specific boundary stations
add_executable(debug_devel_profile src/exec/debug_develop_profile.cpp)

target_link_libraries(debug_devel_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib cases_lib)
target_include_directories(debug_devel_profile PRIVATE src src/physics src/utils src/cases src/data)

###
# Tests
#
enable_testing()

add_subdirectory(tests)