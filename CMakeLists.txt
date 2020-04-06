cmake_minimum_required(VERSION 3.10)

# set the project name
project(team053)

# add the executable
add_executable(denoise_test flt.h tests.hpp flt.c tests.cpp)
