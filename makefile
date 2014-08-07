#CC = clang++
CC = /opt/local/bin/clang++-mp-3.4
O = 3

CFLAGS = -std=c++11 -stdlib=libc++

NOMAD_HOME := $(dir $(firstword $(MAKEFILE_LIST)))
EIGEN_HOME := /usr/local/include

INCLUDES = -I $(NOMAD_HOME) -I $(EIGEN_HOME)

TEST_FILES = $(shell find src/test -type f -name '*_test.cpp')
OBJ_NAMES = $(patsubst src/%.cpp,%.o,$(TEST_FILES))
EXE_NAMES = $(patsubst src/%.cpp,%,$(TEST_FILES))

GTEST ?= lib/gtest_1.7.0
LIBGTEST = test/libgtest.a
GTEST_MAIN = $(GTEST)/src/gtest_main.cc
CFLAGS_GTEST += -I $(GTEST)/include -I $(GTEST)

# GTest Target
$(LIBGTEST): $(LIBGTEST)(test/gtest.o)

test/gtest.o: $(GTEST)/src/gtest-all.cc
	@mkdir -p test
	$(CC) -c -O$O $(CFLAGS_GTEST) $< $(OUTPUT_OPTION)

# Test Object Target
test/%.o : src/test/%.cpp $(LIBGTEST)
	@mkdir -p $(dir $@)
	$(CC) -c -O$O $(CFLAGS_GTEST) $(CFLAGS) $(INCLUDES) $< $(OUTPUT_OPTION)

# Test Executable Target
test/% : test/%.o
	$(CC) -g -O$O $(GTEST_MAIN) $< $(OUTPUT_OPTION) -Ltest -lgtest

scalar: $(EXE_NAMES)

clean:
	rm -rf test/*
