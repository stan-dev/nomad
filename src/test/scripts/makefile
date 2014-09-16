#CC := clang++
CC = /opt/local/bin/clang++-mp-3.6
O := 3

#CLFAGS := -std=c++11
CFLAGS := -std=c++11 -stdlib=libc++

NOMAD_DIR := $(shell dirname $(shell dirname $(shell dirname $(CURDIR))))
EIGEN_DIR := /usr/local/include
GTEST_DIR := $(NOMAD_DIR)/lib/gtest_1.7.0

INCLUDES := -I$(NOMAD_DIR) -I$(EIGEN_DIR)

# GoogleTest
LIBGTEST := $(NOMAD_DIR)/test/libgtest.a
GTEST_MAIN := $(GTEST_DIR)/src/gtest_main.cc
CFLAGS_GTEST := -I$(GTEST_DIR)/include -I$(GTEST_DIR) $(CFLAGS)

$(LIBGTEST): $(LIBGTEST)($(NOMAD_DIR)/test/gtest.o)

$(NOMAD_DIR)/test/gtest.o: $(GTEST_DIR)/src/gtest-all.cc
	@mkdir -p $(NOMAD_DIR)/test
	$(CC) -O$O $(CFLAGS_GTEST) -c $< $(OUTPUT_OPTION)

# Nomad
TESTS := $(shell find $(NOMAD_DIR)/src/test -type f -name '*_test.cpp')
EXES := $(patsubst $(NOMAD_DIR)/src/%.cpp,$(NOMAD_DIR)/%,$(TESTS))

$(NOMAD_DIR)/test/%.o : $(NOMAD_DIR)/src/test/%.cpp $(LIBGTEST)
	@mkdir -p $(dir $@)
	$(CC) -O$O $(CFLAGS_GTEST) $(CFLAGS) $(INCLUDES) -c $< $(OUTPUT_OPTION)

$(NOMAD_DIR)/test/% : $(NOMAD_DIR)/test/%.o
	$(CC) -O$O $(GTEST_MAIN) $< $(CFLAGS_GTEST) $(CFLAGS) $(OUTPUT_OPTION) $(LIBGTEST)

test: $(EXES)

.PHONY: clean
clean:
	rm -rf $(NOMAD_DIR)/test/*
