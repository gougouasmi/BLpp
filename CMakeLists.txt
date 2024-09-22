cmake_minimum_required(VERSION 3.21)

project("Boundary Layer" VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to enable debugging
option(ENABLE_DEBUG "Enable debugging information" OFF)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-march=native)
  message(STATUS "'-march=native' flag added.\n")
endif()

if(ENABLE_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  message(STATUS "Debugging enabled.\n")
endif()

###
# 3rd party libraries
#

find_package(HDF5 REQUIRED COMPONENTS CXX)
message(STATUS "Don't forget to provide HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS} to targets")
message(STATUS "Same for HDF5_LIBRARIES=${HDF5_LIBRARIES}\n")

###
# Configure files
#

# Manually find the individual CSV files
find_file(FLAT_NOSED_PATH "flat_nosed_flow.csv" PATHS ${CMAKE_SOURCE_DIR}/data)
find_file(FLAT_NOSED_PATH_RO "flat_nosed_flow_constant_density.csv" PATHS ${CMAKE_SOURCE_DIR}/data)
find_file(FLAT_NOSED_PATH_RO_COARSE "flat_nosed_flow_constant_density_coarse.csv" PATHS ${CMAKE_SOURCE_DIR}/data)

# Configure the template file with all the CSV file paths at once
configure_file(
    ${CMAKE_SOURCE_DIR}/src/cases/case_functions.h.in
    ${CMAKE_SOURCE_DIR}/src/cases/case_functions.h
    @ONLY
)

###
# Define libraries
#

# Helper functions
add_library(utils_lib
    src/utils/utils.cpp
    src/utils/file_io.cpp
)
target_include_directories(utils_lib PUBLIC src/utils ${HDF5_INCLUDE_DIRS})
target_link_libraries(utils_lib PRIVATE ${HDF5_LIBRARIES})

# Profile functions
add_library(profile_lib
    src/boundary_layer/profile_functions_default.cpp
    src/boundary_layer/profile_functions_cpg.cpp
    src/boundary_layer/score_functions.cpp
)
target_include_directories(profile_lib PRIVATE src/boundary_layer src/physics)
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
    src/boundary_layer/boundary_layer.cpp
    src/boundary_layer/boundary_layer_factory.cpp
)
target_include_directories(boundary_layer_lib PRIVATE src/boundary_layer src/physics src/solvers src/linalg)
target_link_libraries(boundary_layer_lib PRIVATE utils_lib)
target_link_libraries(boundary_layer_lib PRIVATE profile_lib)

# Case setup functions
add_library(cases_lib src/cases/case_functions.cpp)
target_include_directories(cases_lib PRIVATE src/boundary_layer src/physics src/solvers src/linalg src/cases)
target_link_libraries(cases_lib PRIVATE utils_lib)

###
# Define executables
#

# Develop a self-similar profile for set initial conditions
add_executable(devel_profile
    src/exec/develop_profile.cpp
)

target_link_libraries(devel_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(devel_profile PRIVATE src/boundary_layer src/physics src/linalg)

# Search for the initial conditions that develop into the correct self-similar profile
add_executable(search_profile
    src/exec/search_profile.cpp
)

target_link_libraries(search_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib)
target_include_directories(search_profile PRIVATE src/boundary_layer src/physics src/utils)

# Compute 2D profile (local-similarity)
add_executable(compute_2d_profile
    src/exec/compute_2d_profile.cpp
)

target_link_libraries(compute_2d_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib cases_lib)
target_include_directories(compute_2d_profile PRIVATE src/boundary_layer src/physics src/utils src/cases)

# Compute edge flow from pressure
add_executable(edge_solve src/exec/solve_from_pressure.cpp src/solvers/edge_solvers.cpp)

target_link_libraries(edge_solve PRIVATE utils_lib linalg_lib)
target_include_directories(edge_solve PRIVATE src/boundary_layer src/utils src/solvers src/linalg)

# (DEBUGGING) Develop profile for specific boundary stations
add_executable(debug_devel_profile src/exec/debug_develop_profile.cpp)

target_link_libraries(debug_devel_profile PRIVATE boundary_layer_lib profile_lib utils_lib linalg_lib cases_lib)
target_include_directories(debug_devel_profile PRIVATE src/boundary_layer src/physics src/utils src/cases)

###
# Tests
#
enable_testing()

add_subdirectory(tests)