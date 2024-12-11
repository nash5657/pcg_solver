cmake_minimum_required(VERSION 3.10)

# Set the project name
set(PROJECT_NAME pcg_solver)
project(${PROJECT_NAME} CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3
find_package(Eigen3 REQUIRED)
if (NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found")
endif()

# Add an executable
add_executable(${PROJECT_NAME} main.cpp preconditioner.cpp matrix_generator.cpp pcg_solver.cpp)

# Manually specify OpenMP flags and libraries for macOS
if (APPLE)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_INCLUDE_DIR "/opt/homebrew/opt/llvm/include")
    set(OpenMP_LIBRARIES "/opt/homebrew/opt/llvm/lib/libomp.dylib")
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenMP_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenMP_LIBRARIES})

    # Add standard library paths (fix for macOS)
    target_include_directories(${PROJECT_NAME} PRIVATE /Library/Developer/CommandLineTools/usr/include/c++/v1)
    target_include_directories(${PROJECT_NAME} PRIVATE /opt/homebrew/opt/llvm/include/c++/v1)

    # Explicitly set the compiler (if necessary)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
else()
    # Standard OpenMP configuration for non-Apple platforms
    find_package(OpenMP REQUIRED)
    if (NOT OpenMP_FOUND)
        message(FATAL_ERROR "OpenMP not found")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Include Eigen3 and link libraries
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# Diagnostic messages
message(STATUS "Using OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
