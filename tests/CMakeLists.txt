# Check if the tests are being built as a standalone project
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.10)
    project(Tests VERSION 1.0)

    # Specify the C++ standard
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    # Include the src directory
    include_directories(${CMAKE_SOURCE_DIR}/../src)

else()
    # Include the src directory relative to the main project
    include_directories(${CMAKE_SOURCE_DIR}/src)
endif()

enable_testing()

add_executable(test_gas_model gas_model_test.cpp)
add_test(NAME TestGasModel COMMAND test_gas_model)