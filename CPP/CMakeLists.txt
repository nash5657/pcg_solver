cmake_minimum_required(VERSION 3.10)

# Set the project name
set(PROJECT_NAME pcg_solver)
project(${PROJECT_NAME} CXX)  # Use CXX for C++ projects, not C++)


# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


#find package
find_package(Eigen3 REQUIRED)
if (NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found")
endif()


# Add an executable
add_executable(${PROJECT_NAME} main.cpp preconditioner.cpp matrix_generator.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)


