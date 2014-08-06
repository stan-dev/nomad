#include <gtest/gtest.h>

#include <tests/scalar/functions.hpp>

//clang++ -O3 -std=c++11 -I../.. -I/usr/local/include -I/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/lib/gtest_1.7.0/include -L/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/test -lgtest -o run_scalar_function_tests run_scalar_function_tests.cpp

clang++ -O3 -std=c++11 -I/Users/Betancourt/Documents/Research/Code/stan-dev/nomad -I/usr/local/include -I/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/lib/gtest_1.7.0/include -L/Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/test -lgtest -o acos_test acos_test.cpp /Users/Betancourt/Documents/Research/Code/stan-dev/cmdstan/stan/lib/gtest_1.7.0/src/gtest_main.cc

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}