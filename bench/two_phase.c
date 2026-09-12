/*
 * two_phase.c -- a two-phase OpenMP benchmark: memory-bound, then compute-bound.
 *
 * The workload the phase-detection autoencoder and the full control loop were
 * validated on (PhD thesis section 4.3.2.2, Figs. 4.23-4.24). It alternates
 * between two long stretches derived from the extremes of mixed_cxmy.c, and the
 * two stretches have deliberately *distant* optimal configurations:
 *
 *   memory-bound phase  -> best around  4 cores @ 1.3 GHz
 *   compute-bound phase -> best around 19 cores @ 2.2 GHz
 *
 * That distance is the point. A controller with no phase input sees only a CpS
 * value, and CpS ranges from the two phases overlap once the configuration also
 * varies -- so it cannot tell them apart and settles on a compromise. Given the
 * autoencoder's phase code instead, the same controller tracks the core count
 * almost exactly (19 and 5 against a known-best 19 and 4) and gains 34% mean CpJ
 * over the phase-free version.
 *
 * Contrast with SRAD (Rodinia), whose two phases have *similar* optimal
 * configurations: there the controller detects both phases but does not
 * meaningfully separate their configurations, because there is little to gain by
 * doing so. Both results are in thesis section 4.3.2.
 *
 *   usage: ./two_phase <iterations> <mem_phase_iters> <cpu_phase_iters> <nthreads>
 *   e.g.:  ./two_phase 1000000000 5000000 20000000 19
 *
 * The argument order and meaning are inferred from the surviving invocation in
 * the original launch scripts; the original C source did not survive (see
 * docs/provenance.md and bench/README.md).
 *
 * As with mixed_cxmy.c, schedule(dynamic, 1) makes one chunk equal one outer
 * iteration, and chunk counting requires the patched GCC/libgomp
 * (docs/libgomp-instrumentation.md).
 *
 * Build:  make          (or: gcc -O2 -fopenmp -o two_phase two_phase.c -lm)
 */

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define INNER 100
#define MEM_DOUBLES (16u * 1024u * 1024u)
#define MEM_MASK (MEM_DOUBLES - 1u)
#define FOURIER_TERMS 24

static double *mem_a;
static double *mem_b;
static volatile double sink = 0.0;

static inline uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline double mem_fragment(uint64_t *state)
{
    uint32_t i0 = (uint32_t)(xorshift64(state) & MEM_MASK);
    uint32_t i1 = (uint32_t)(xorshift64(state) & MEM_MASK);
    uint32_t i2 = (uint32_t)(xorshift64(state) & MEM_MASK);

    double a = mem_a[i0];
    double b = mem_b[i1];

    mem_b[i2] = a + b;
    mem_a[i1] = b - a;

    return a * 0.5 + b * 0.25;
}

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
            "usage: %s <iterations> <mem_phase_iters> <cpu_phase_iters> <nthreads>\n"
            "  iterations       total outer iterations; one iteration = one chunk\n"
            "  mem_phase_iters  iterations spent in the memory-bound phase\n"
            "  cpu_phase_iters  iterations spent in the compute-bound phase\n"
            "  nthreads         OpenMP threads\n"
            "The two phases alternate until <iterations> is reached.\n",
            program);
}

int main(int argc, char **argv)
{
    long long iterations, mem_phase, cpu_phase, period;
    int nthreads;
    unsigned int i;

    if (argc != 5) {
        usage(argv[0]);
        return 1;
    }
    iterations = atoll(argv[1]);
    mem_phase = atoll(argv[2]);
    cpu_phase = atoll(argv[3]);
    nthreads = atoi(argv[4]);

    if (iterations <= 0 || mem_phase <= 0 || cpu_phase <= 0 || nthreads <= 0) {
        usage(argv[0]);
        return 1;
    }
    period = mem_phase + cpu_phase;

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
            "two_phase: %lld iterations, phases mem=%lld / cpu=%lld, %d threads\n",
            iterations, mem_phase, cpu_phase, nthreads);

    /* The phase is a function of the iteration index, not of wall time, so the
     * phase boundaries stay at the same point in the work regardless of how many
     * cores or what frequency the controller picks. The controller therefore
     * cannot shift a phase boundary by acting -- which is what makes the phases a
     * property of the workload rather than of the control policy. */
    double started = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1)
    for (long long n = 0; n < iterations; ++n) {
        int memory_phase = (n % period) < mem_phase;
        uint64_t state = (uint64_t)n * 6364136223846793005ULL + 1442695040888963407ULL;
        double local = 0.0;
        int j;
        for (j = 0; j < INNER; ++j) {
            if (memory_phase) {
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
