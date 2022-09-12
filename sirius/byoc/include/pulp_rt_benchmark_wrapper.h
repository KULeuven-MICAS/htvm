#ifndef PULP_RT_BENCHMARK_WRAPPER_H
#define PULP_RT_BENCHMARK_WRAPPER_H

#include <pulp.h>

// initialization of global performance counter
volatile rt_perf_t *perf;

void __attribute__((noinline, optimize("O0"))) init_global_perf_counter();

void __attribute__((noinline, optimize("O0"))) start_benchmark();

int32_t __attribute__((noinline, optimize("O0"))) stop_benchmark();

#endif // PULP_RT_BENCHMARK_WRAPPER_H
