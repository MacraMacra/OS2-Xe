# SPDX-License-Identifier: MIT
#
#   source /opt/intel/oneapi/setvars.sh
#   make -j                 tests, examples and benchmarks, AOT for Arc B-series (bmg-g21)
#   make -j AOT=0           JIT build (SPIR-V, any Intel GPU)
#   make check              run the correctness tests
#   make check-quick        run the tests without the large cases

ifeq ($(origin CXX),default)
CXX = icpx
endif
AOT      ?= 1
AOT_DEV  ?= bmg-g21
CXXFLAGS ?= -O3 -fp-model=precise
SYCLFLAGS = -fsycl -fsycl-device-code-split=per_kernel
ifeq ($(AOT),1)
SYCLFLAGS += -fsycl-targets=spir64_gen -Xs "-device $(AOT_DEV)"
endif
INCLUDES = -Isrc
LIBS     = -qmkl -qopenmp
BUILD   ?= build

HDRS     = src/ozaki2_gpu.hpp src/coprime_tables.hpp
TESTS    = $(BUILD)/test_ozaki2
EXAMPLES = $(BUILD)/dgemm $(BUILD)/zgemm $(BUILD)/profiling
BENCHES  = $(BUILD)/bench_ozaki2 $(BUILD)/bench_accuracy

all: tests examples benchmarks
tests: $(TESTS)
examples: $(EXAMPLES)
benchmarks: $(BENCHES)

$(BUILD)/%: tests/%.cpp $(HDRS) | $(BUILD)
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BUILD)/%: examples/%.cpp $(HDRS) | $(BUILD)
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BUILD)/%: benchmarks/%.cpp benchmarks/ozaki1_gpu.hpp $(HDRS) | $(BUILD)
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BUILD):
	mkdir -p $@

check: $(TESTS)
	ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./$(BUILD)/test_ozaki2

check-quick: $(TESTS)
	ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./$(BUILD)/test_ozaki2 --quick

clean:
	rm -rf $(BUILD)

.PHONY: all tests examples benchmarks check check-quick clean
