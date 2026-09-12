/*
 * mixed_cxmy.c -- parametrizable CPU/memory-intensity OpenMP benchmark.
 *
 * The benchmark template of PhD thesis Fig. 4.12 / MOCAST 2020 Fig. 2. One outer
 * loop iteration is split into INNER inner steps; a `coef` parameter decides how
 * many of those steps run a memory-intensive fragment and how many run a
 * compute-intensive one. Varying that one number sweeps a workload continuously
 * from purely memory-bound to purely compute-bound, which is the point: the two
 * ends of that spectrum have *opposite* optimal configurations -- one core for
 * the memory-bound end, all 19 for the compute-bound end -- and a controller
 * that gets both right has demonstrably learned something about the workload
 * rather than about the machine.
 *
 * The six benchmarks characterized in the thesis (Table 4.2) are this program at
 * six settings: C100M0, C98M2, C96M4, C90M10, C80M20, C0M100.
 *
 *   usage: ./mixed_cxmy <iterations> <cpu_percent> <nthreads>
 *   e.g.:  ./mixed_cxmy 2000000000 100 19     # C100M0, purely compute-bound
 *          ./mixed_cxmy 2000000000  90 19     # C90M10
 *          ./mixed_cxmy 2000000000   0 19     # C0M100, purely memory-bound
 *
 * Why schedule(dynamic, 1): one chunk must equal exactly one outer iteration, so
 * that the chunk counter the patched libgomp maintains is a direct measure of
 * work done. A larger chunk size, or static scheduling, would make the counter
 * coarser than the 500 ms control period and the CpS signal useless. This is
 * what the "_1chunk" suffix meant in the original binaries' names.
 *
 * Chunk counting requires linking against the patched GCC/libgomp
 * (docs/libgomp-instrumentation.md). Without it this still builds and runs -- it
 * is simply not instrumented, and nothing publishes a chunk counter.
 *
 * Build:  make          (or: gcc -O2 -fopenmp -o mixed_cxmy mixed_cxmy.c -lm)
 *
 * This is a reconstruction from the published template plus the command-line
 * contract recovered from the original launch scripts; the original C source did
 * not survive (see docs/provenance.md).
 */

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Inner steps per outer iteration. The template fixes this at 100 so that
 * `cpu_percent` reads directly as a percentage. */
#define INNER 100

/* Working set for the memory fragment: 2 x 128 MiB. Deliberately far larger
 * than any last-level cache on the target class of machine (the Xeon E5-2630 v4
 * has 25 MiB of L3 per socket), so the fragment is bound by DRAM bandwidth and
 * latency rather than by cache. */
#define MEM_DOUBLES (16u * 1024u * 1024u)
#define MEM_MASK (MEM_DOUBLES - 1u)

/* Terms in the compute fragment's Fourier partial sum. Enough transcendental
 * work per step that the fragment stays register-resident and CPU-bound. */
#define FOURIER_TERMS 24

static double *mem_a;
static double *mem_b;

/* Written but never read back, so the compiler cannot elide the work; volatile
 * keeps the stores. Racy across threads by design -- it is a sink, not a
 * result. */
static volatile double sink = 0.0;

/* Cheap per-thread xorshift, used to scatter memory accesses. A sequential
 * access pattern would be caught by the hardware prefetcher and would stop the
 * "memory-intensive" fragment from actually stressing memory. */
static inline uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Memory-intensive fragment: scattered read, arithmetic, scattered write.
 * Three streams touching unrelated cache lines per step. */
static inline double mem_fragment(uint64_t *state)
{
    uint32_t i0 = (uint32_t)(xorshift64(state) & MEM_MASK);
    uint32_t i1 = (uint32_t)(xorshift64(state) & MEM_MASK);
    uint32_t i2 = (uint32_t)(xorshift64(state) & MEM_MASK);

    double a = mem_a[i0];
    double b = mem_b[i1];

    mem_b[i2] = a + b;          /* write: dirties a line, forces write-back  */
    mem_a[i1] = b - a;          /* permute: the two arrays feed each other   */

    return a * 0.5 + b * 0.25;
}

/* Compute-intensive fragment: a Fourier partial sum plus a short chain of
 * dependent floating-point operations. Touches no memory beyond registers. */
static inline double cpu_fragment(double t)
{
    double acc = 0.0;
    int k;
    for (k = 1; k <= FOURIER_TERMS; ++k) {
        double w = (double)(2 * k - 1);
        acc += sin(w * t) / w + cos(w * t) / (w * w);
    }
    acc = sqrt(fabs(acc) + 1.0);
    acc = acc * acc + log(acc + 1.0);
    return acc;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s <iterations> <cpu_percent> <nthreads>\n"
            "  iterations   outer loop iterations; one iteration = one OpenMP chunk\n"
            "  cpu_percent  0..100 -- share of inner steps spent compute-bound\n"
            "               (100 => C100M0, 90 => C90M10, 0 => C0M100)\n"
            "  nthreads     OpenMP threads\n",
            program);
}

int main(int argc, char **argv)
{
    long long iterations;
    int cpu_percent, nthreads, mem_threshold;
    unsigned int i;

    if (argc != 4) {
        usage(argv[0]);
        return 1;
    }
    iterations = atoll(argv[1]);
    cpu_percent = atoi(argv[2]);
    nthreads = atoi(argv[3]);

    if (iterations <= 0 || cpu_percent < 0 || cpu_percent > 100 || nthreads <= 0) {
        usage(argv[0]);
        return 1;
    }

    /* Inner steps below the threshold run the memory fragment, the rest the
     * compute one -- so cpu_percent = 100 leaves no memory steps at all. */
    mem_threshold = 100 - cpu_percent;

    if (posix_memalign((void **)&mem_a, 64, MEM_DOUBLES * sizeof(double)) != 0 ||
        posix_memalign((void **)&mem_b, 64, MEM_DOUBLES * sizeof(double)) != 0) {
        fprintf(stderr, "allocation of 2 x %u MiB failed\n",
                (unsigned)(MEM_DOUBLES * sizeof(double) / (1024 * 1024)));
        return 1;
    }
    for (i = 0; i < MEM_DOUBLES; ++i) {
        mem_a[i] = 0.1;
        mem_b[i] = 0.0;
    }

    omp_set_num_threads(nthreads);
    fprintf(stderr,
            "mixed_cxmy: C%dM%d, %lld iterations, %d threads, %u MiB working set\n",
            cpu_percent, 100 - cpu_percent, iterations, nthreads,
            (unsigned)(2 * MEM_DOUBLES * sizeof(double) / (1024 * 1024)));

    double started = omp_get_wtime();

    /* One chunk = one iteration of this loop. */
#pragma omp parallel for schedule(dynamic, 1)
    for (long long n = 0; n < iterations; ++n) {
        uint64_t state = (uint64_t)n * 6364136223846793005ULL + 1442695040888963407ULL;
        double local = 0.0;
        int j;
        for (j = 0; j < INNER; ++j) {
            if (j < mem_threshold) {
                local += mem_fragment(&state);
            } else {
                local += cpu_fragment((double)j * 0.01 + (double)(n & 0xFF));
            }
        }
        sink = local;
    }

    {
        /* Reported on stderr so it never contaminates a workload's own
         * stdout. Excludes allocation and first-touch of the working set,
         * which otherwise dominate short runs. */
        double elapsed = omp_get_wtime() - started;
        fprintf(stderr, "  %lld chunks in %.3f s = %.0f chunks/s\n",
                iterations, elapsed, (double)iterations / elapsed);
    }

    free(mem_a);
    free(mem_b);
    return 0;
}
