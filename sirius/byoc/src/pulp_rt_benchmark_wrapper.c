#include <pulp.h>
#include <pulp_rt_benchmark_wrapper.h>


void  __attribute__((noinline, optimize("O0"))) init_global_perf_counter(){
    perf = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(rt_perf_t));
}

void __attribute__((noinline, optimize("O0"))) start_benchmark(){
    // perf is globally defined in pulp_rt_benchmark_wrapper.h
    rt_perf_init(perf);
    rt_perf_reset(perf);
    rt_perf_conf(perf, (1<<RT_PERF_CYCLES));
    rt_perf_stop(perf);
    rt_perf_start(perf);
}

int32_t __attribute__((noinline, optimize("O0"))) stop_benchmark(){
    // perf is globally defined in pulp_rt_benchmark_wrapper.h
    rt_perf_stop(perf);
    rt_perf_save(perf);
    int32_t perf_cyc = rt_perf_get(perf, RT_PERF_CYCLES);
    rt_perf_reset(perf);
    return perf_cyc;
}
