#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <chrono>

#include <stdio.h>
#include <random>
#include <tuple>
#include <math.h>
/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>
#include <curand_kernel.h>

//#define SAMPLES_PER_THREAD 100000
// Samples per thread should be at most
// 10^3 for 10^6 samples
// 10^4 for 10^7 samples
// 10^5 for 10^8 samples
 
#define BLOCK_SIZE 256
#define BLOCKS_PER_STREAM 1
#define MAX_STREAMS_COUNT 64

// CPU CODE FOR STD TUNING 

float cpu_norm_pdf(float mean, float std, float x) {
    const float a = 1 / sqrt(2 * M_PI);
    return a / std * exp(-0.5 * pow(((x - mean) / std), 2));
}

float cpu_ff(float x) {
    x = x / 100.0;
    return cpu_norm_pdf(0.0, 1.0, x) + cpu_norm_pdf(3.0, 1.0, x) + cpu_norm_pdf(6.0, 1.0, x);
}

std::tuple<float, float> cpu_burn_loop(float xt, float std, int burn_N, float (*f)(float), std::mt19937 gen) {
    float f_xt = f(xt);
    std::uniform_real<float> uni(0, 1);
    const float target = 0.3;
    int accepted = 0;



    for (int t = 0; t < burn_N; t++) {
        std::normal_distribution<> g(xt, std);
        float xc = g(gen);
        float f_xc = f(xc);
        float a = f_xc / f_xt;
        float u = uni(gen);
        if (u <= a) {
            xt = xc;
            f_xt = f_xc;
            accepted += 1;
        }
        float temp = accepted / (t + 1.0);
        float reg = 1 / std * 1000.0 * (temp - target);
        if (!(reg < 0 && abs(reg) > std)) {
            std += reg;
        }
    }
    return std::tuple<float, float>(xt, std);
}


// DEVICE CODE

__device__ float dev_norm_pdf(float mean, float std, float x) {
    const float inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi / std * exp(-pow((x - mean) / std, 2) / 2.0);
}

__device__ float dev_ff(float x) {
    x = x / 100.0;
    return dev_norm_pdf(0.0, 1.0, x) + dev_norm_pdf(3.0, 1.0, x) + dev_norm_pdf(6.0, 1.0, x);
}

__device__ float(*dev_ff_p)(float) = dev_ff;


__global__ void dev_generate(curandStateMtgp32* state, float* x, int x_len, float xt, float std,int samples_per_thread, float (*f)(float)) {
    int warp_id = threadIdx.x / 32;
    int id_in_warp = threadIdx.x % 32;
    int warp_range_start = samples_per_thread * BLOCK_SIZE * blockIdx.x + warp_id * 32 * samples_per_thread;
    int warp_range_end = samples_per_thread * BLOCK_SIZE * blockIdx.x + (warp_id + 1) * 32 * samples_per_thread;

    if (warp_range_end > x_len)
        warp_range_end = x_len;

    float f_xt = f(xt);

    // new starting point for each thread
    for (int t = 0; t < 1000; t++) {
        float xc = xt + curand_normal(&state[blockIdx.x]) * std / 4.0f ;
        float f_xc = f(xc);
        float a = f_xc / f_xt;
        float u = curand_uniform(&state[blockIdx.x]);

        if (u <= a) {
            xt = xc;
            f_xt = f_xc;
        }
    }

    for (int t = warp_range_start; t < warp_range_end  ; t+=32) {
        float xc = xt + curand_normal(&state[blockIdx.x])*std ;
        float f_xc = f(xc);
        float a = f_xc / f_xt;
        float u = curand_uniform(&state[blockIdx.x]);
  
        if (u <= a) {
            xt = xc;
            f_xt = f_xc;
        }
        x[t + id_in_warp] = xt;
        // W tej pętli każdy wątek w ramach warpu pisze do co 32 adresu w pamięci, żeby lepiej grupować 
        // transfery pamięci
        // TODO: Do porównania z poniższą zakomentowaną wersją, gdzie każdy wątek pisze do swojego przedziału
    }


    // int i = SAMPLES_PER_THREAD * BLOCK_SIZE * blockIdx.x   + threadIdx.x * SAMPLES_PER_THREAD;
    // int sstart_range = i ;
    // int eend_range= i + SAMPLES_PER_THREAD;
    // float f_xt = f(xt);
    // if (eend_range > x_len) { eend_range = x_len; }
    // for (int t = sstart_range; t < eend_range  ; t++) {
    //     float xc = xt + curand_normal(&state[blockIdx.x])*std ;
    //     float f_xc = f(xc);
    //     float a = f_xc / f_xt;
    //     float u = curand_uniform(&state[blockIdx.x]);
  
    //     if (u <= a) {
    //         xt = xc;
    //         f_xt = f_xc;
    //     }
    //     x[t] = xt;
    // }
}

int mh(float* x, float x0, int N, int burn_N, float (*cpu_f)(float), float (*f)(float), int seed) {
    
   /* int temp = N;
    int p = 0;
    while (temp > 0) {
        temp = temp / 10;
        p++;
    }
    p = pow(10, p - 4);
    p = std::min(p, 100000);
    const int samples_per_thread = std::max(p, 1000);*/
    const int samples_per_thread = 10000;

    const int samples_per_block = (BLOCK_SIZE * samples_per_thread);
    const int block_count = (N + samples_per_block - 1) / samples_per_block;
    const int stream_count = 1 + (block_count - 1) / BLOCKS_PER_STREAM;
    const int samples_per_stream = BLOCKS_PER_STREAM * BLOCK_SIZE * samples_per_thread;


    typedef std::chrono::high_resolution_clock Clock;
    cudaEvent_t start, stop;
    float elapsedTime;
    Clock::time_point t0 = Clock::now();

    checkCudaErrors(cudaSetDevice(0));

    float* dev_x;

    size_t free = 0, total= 0;
    cudaMemGetInfo(&free, &total);

    const int MEMORY_POOLS_AVAILABLE = free / sizeof(float) / samples_per_stream;
     int MEMORY_POOLS = MEMORY_POOLS_AVAILABLE * 8 / 10;

    if (stream_count < MEMORY_POOLS) MEMORY_POOLS = stream_count;

    cudaHostRegister(x, N * sizeof(float), 0);


    // Strojenie odchylenia standardowego wspólne na CPU
    float xt = x0;
    float std = 1.0;
    std::mt19937 gen(seed);
    std::tuple<float, float> res;
    res = cpu_burn_loop(xt, std, burn_N, cpu_f, gen);
    //printf("std = %f \n", std::get<1>(res));

    xt = std::get<0>(res);
    std = std::get<1>(res);




    float (*fff)(float);
    checkCudaErrors(cudaMemcpyFromSymbol(&fff, dev_ff_p, sizeof(f)));
    // TODO: tutaj jest problem, że nie można podać do cudaMemcpyFromSymbol na drugi arg
    // funkcji f, która została podana do aktualnej funkcji - coś z tym zrobić trzeba


    int *samples_pool;
    cudaHostAlloc((void**)&samples_pool, stream_count * sizeof(int), 0);
    int temp_N = N;
    for (int i = 0; i < stream_count; i++) {
        if (temp_N >= samples_per_stream) {
            samples_pool[i] = samples_per_stream;
            temp_N -= samples_per_stream;
        }
        else {
            samples_pool[i] = temp_N;
        }
    }


   
    checkCudaErrors(cudaMalloc((void**)&dev_x, MEMORY_POOLS * samples_per_stream * sizeof(float)));


    cudaStream_t *stream;
    cudaEvent_t *streamMemoryPoolFreeEvent;

    checkCudaErrors(cudaHostAlloc((void**)&stream, stream_count * sizeof(cudaStream_t), 0));
    checkCudaErrors(cudaHostAlloc((void**)&streamMemoryPoolFreeEvent , stream_count * sizeof(cudaEvent_t), 0));
    
    for (int i = 0; i < stream_count; i++) {
        checkCudaErrors(cudaStreamCreateWithFlags(stream + i, cudaStreamNonBlocking));
        cudaEventCreateWithFlags(streamMemoryPoolFreeEvent + i, cudaEventDisableTiming & cudaEventBlockingSync);
    }

    // Używamy generatora MTGP32, mamy do porównania wersje zaimplementowane na CPU sekwencyjnie i równolegle i
    // one wykorzystują również ten sam generator
    curandStateMtgp32* devMTGPStates;
    mtgp32_kernel_params* devKernelParams;

    // Każdy blok dostaje swój generator, każdy generator może wydajnie obsłużyć max 256 wątków w ramach bloku
    checkCudaErrors(cudaMalloc((void**)&devMTGPStates, block_count * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));

    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);

    /* Initialize one state per thread block */
    //curandMakeMTGP32KernelState(devMTGPStates,mtgp32dc_params_fast_11213, devKernelParams, block_num, seed * 117);
    for (int i = 0; i < block_count; i++) {
        curandMakeMTGP32KernelState(devMTGPStates + i, mtgp32dc_params_fast_11213, devKernelParams, 1, seed * 7 * i);
    }
    
   
    for (int s = 0; s < stream_count; s++) {
        //if (s >= MEMORY_POOLS) checkCudaErrors(cudaEventSynchronize(streamMemoryPoolFreeEvent[s]));
        if (s >= MEMORY_POOLS) checkCudaErrors(cudaStreamWaitEvent(stream[s], streamMemoryPoolFreeEvent[s]));
        dev_generate<<<BLOCKS_PER_STREAM, BLOCK_SIZE, 0, stream[s]>>>(devMTGPStates + s * BLOCKS_PER_STREAM, dev_x + s % MEMORY_POOLS * samples_per_stream, samples_pool[s], xt, std,samples_per_thread, fff);
        checkCudaErrors(cudaMemcpyAsync(x + s * samples_per_stream, dev_x + (s % MEMORY_POOLS) * samples_per_stream, samples_pool[s] * sizeof(float), cudaMemcpyDeviceToHost, stream[s]));
        if(s+MEMORY_POOLS < stream_count) checkCudaErrors(cudaEventRecord(streamMemoryPoolFreeEvent[(s+MEMORY_POOLS)], stream[s]));
    }


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    for (int s = 0; s < stream_count; s++) {
        checkCudaErrors(cudaStreamDestroy(stream[s]));
    }
    Clock::time_point t1 = Clock::now();

    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    // for kernel time measurement
    // printf("Kernel-only time: %f ms\n", elapsedTime);
    printf("All time %d ms \n", d.count());
 
    checkCudaErrors(cudaFree(dev_x));
    cudaHostUnregister(x);

    return 0;
}

extern "C" float* cuda_main(int N, int burn_N)
{
    float* x = (float*)malloc(sizeof(float) * N);

    mh(x, 0.0, N, burn_N, cpu_ff,dev_ff_p, 7);

    checkCudaErrors(cudaDeviceReset());

    return x;
}

