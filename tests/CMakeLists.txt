# Check if the tests are being built as a standalone project
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.10)
    project(Tests VERSION 1.0)

    # Specify the C++ standard
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    # Include the src directory
    include_directories(../src ../src/physics)
else()
    # Include the src directory relative to the main project
    include_directories(${CMAKE_SOURCE_DIR}/src)
    include_directories(${CMAKE_SOURCE_DIR}/src/physics)
endif()

enable_testing()

add_executable(test_gas_model gas_model_test.cpp)
add_test(NAME TestGasModel COMMAND test_gas_model)

add_executable(test_atmosphere atmosphere_test.cpp)
add_test(NAME TestAtmosphere COMMAND test_atmosphere)

#message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}\n")
#message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}\n")
#get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
#message(STATUS "Include directories: ${inc_dirs}\n")