#include <gtest/gtest.h>

#include <tests/scalar/functions.hpp>

//clang++ -O3 -std=c++11 -I../.. -I/usr/local/include -I/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/lib/gtest_1.7.0/include -L/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/test -lgtest -o run_scalar_function_tests run_scalar_function_tests.cpp

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}