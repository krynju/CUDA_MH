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

#define SAMPLES_PER_JOB 10000
#define BLOCK_SIZE 250


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


__global__ void dev_generate(curandStateMtgp32* state, float* x, int x_len, float xt, float std, float (*f)(float)) {
    int i = SAMPLES_PER_JOB * BLOCK_SIZE * blockIdx.x   + threadIdx.x * SAMPLES_PER_JOB;
    int sstart_range = i ;
    int eend_range= i + SAMPLES_PER_JOB;

    if (eend_range > x_len) { eend_range = x_len; }
  
    float f_xt = f(xt);
    for (int t = sstart_range; t < eend_range; t++) {
        float xc = xt + curand_normal(&state[blockIdx.x])*std ;
        float f_xc = f(xc);
        float a = f_xc / f_xt;
        float u = curand_uniform(&state[blockIdx.x]);
  
        if (u <= a) {
            xt = xc;
            f_xt = f_xc;
        }
        x[t] = xt;
    }
}

int mh(float*x,float x0, int N, int burn_N, float (*cpu_f)(float), float (*f)(float), int seed) {
    const int samples_per_block = (BLOCK_SIZE * SAMPLES_PER_JOB);
    const int block_num = (N + samples_per_block - 1) / samples_per_block;

    typedef std::chrono::high_resolution_clock Clock;
    
    cudaEvent_t start, stop;
    float elapsedTime;
    Clock::time_point t0 = Clock::now();

    checkCudaErrors(cudaSetDevice(0));

    float* dev_x;

    checkCudaErrors(cudaMalloc((void**)&dev_x, N * sizeof(float)));

    
    float xt = x0;
    float std = 1.0;

    std::mt19937 gen(seed);

    std::tuple<float, float> res;
    res = cpu_burn_loop( xt, std, burn_N, cpu_f, gen);
    //printf("std = %f \n", std::get<1>(res));

    xt = std::get<0>(res);
    std = std::get<1>(res);

    curandStateMtgp32* devMTGPStates;
    mtgp32_kernel_params* devKernelParams;

    checkCudaErrors(cudaMalloc((void**)&devMTGPStates, block_num * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params)));

    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);

    /* Initialize one state per thread block */
    curandMakeMTGP32KernelState(devMTGPStates,mtgp32dc_params_fast_11213, devKernelParams, block_num, seed * 117);

    float (*fff)(float);
    checkCudaErrors(cudaMemcpyFromSymbol(&fff, dev_ff_p, sizeof(f))); //tu nie mogę podać f po prostu tylko muszę ff_gpu_stat


    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

   
    dev_generate<<<block_num,BLOCK_SIZE>>>(devMTGPStates, dev_x, N, xt, std, fff);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));


    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(x, dev_x, N * sizeof(float), cudaMemcpyDeviceToHost));
  

    Clock::time_point t1 = Clock::now();

    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    printf("Kernel-only time: %f ms\n", elapsedTime);
    printf("All time %d ms \n", d.count());
 
    checkCudaErrors(cudaFree(dev_x));

    return 0;
}

extern "C" float* cuda_main(int N, int burn_N)
{
    float* x = (float*)malloc(sizeof(float) * N);

    mh(x, 0.0, N, burn_N, cpu_ff,dev_ff_p, 1117);

    checkCudaErrors(cudaDeviceReset());

    return x;
}

