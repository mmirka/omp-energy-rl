# Builds the two instrumented example applications.
#
# Chunk counting -- the whole point of these programs -- only happens when they
# are linked against the patched GCC/libgomp described in
# docs/libgomp-instrumentation.md. Point CC at that toolchain to get an
# instrumented binary:
#
#     make CC=/opt/gcc-4.8.0-chunkcount/bin/gcc
#
# With a stock compiler they build and run identically, just uninstrumented.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -fopenmp
LDLIBS  ?= -lm

BINARIES = mixed_cxmy two_phase

all: $(BINARIES)

mixed_cxmy: mixed_cxmy.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

two_phase: two_phase.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# A few seconds each, just to confirm they build and run.
smoke: $(BINARIES)
	./mixed_cxmy 20000 100 4
	./mixed_cxmy 20000 0 4
	./two_phase 20000 500 2000 4

clean:
	rm -f $(BINARIES)

.PHONY: all smoke clean
