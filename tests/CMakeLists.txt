# Check if the tests are being built as a standalone project
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.10)
    project(Tests VERSION 1.0)

    # Specify the C++ standard
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    # Include the src directory
    include_directories(../src/boundary_layer ../src/physics ../src/data)
else()
    # Include the src directory relative to the main project
    include_directories(${CMAKE_SOURCE_DIR}/src/boundary_layer)
    include_directories(${CMAKE_SOURCE_DIR}/src/physics)
    include_directories(${CMAKE_SOURCE_DIR}/src/data)
endif()

enable_testing()

# Test gas model functions
add_executable(test_gas_model gas_model_test.cpp)
add_test(NAME TestGasModel COMMAND test_gas_model)

# Test atmosphere functions
add_executable(test_atmosphere atmosphere_test.cpp)
add_test(NAME TestAtmosphere COMMAND test_atmosphere)

# Test direct LU solve
add_executable(test_dense_solve
    dense_linear_solve_test.cpp
    ../src/linalg/dense_direct_solver.cpp
    ../src/linalg/dense_linalg.cpp
)
target_include_directories(test_dense_solve PRIVATE ../src/linalg)
add_test(NAME TestDenseSolver COMMAND test_dense_solve)

# Test Newton solve
add_executable(test_newton_solve
    newton_solve_test.cpp 
    ../src/linalg/dense_direct_solver.cpp
    ../src/linalg/dense_matrix.cpp
)
target_include_directories(test_newton_solve PRIVATE ../src/linalg ../src/solvers)
add_test(NAME TestNewtonSolver COMMAND test_newton_solve)

# Test profile functions 
add_executable(test_profile_functions
    profile_functions_test.cpp 
    ../src/boundary_layer/profile_functions_default.cpp
    ../src/boundary_layer/profile_functions_cpg.cpp
)
#target_include_directories(test_profile_functions PRIVATE <include directories>)
add_test(NAME TestProfileFunctions COMMAND test_profile_functions)

# Test shock calculator
add_executable(test_shock_functions
  shock_test.cpp
)
add_test(NAME TestShockFunctions COMMAND test_shock_functions)

# Test stagnation ratio calculator
add_executable(test_stagnation stagnation_test.cpp)
add_test(NAME TestStagnation COMMAND test_stagnation)