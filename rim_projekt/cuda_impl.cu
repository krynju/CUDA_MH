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
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


#define PER_JOB 10000
#define BLOCK_SIZE 250
#define NN 10000000/4

double norm_pdf(double mean, float std, float x) {
    const double a = 1 / sqrt(2 * M_PI);
    return a / std * exp(-0.5 * pow(((x - mean) / std), 2));
}

__device__ float norm_pdf_gpu(float mean, float std, float x) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - mean) / std;
    float b = exp(-pow(a, 2) / 2.0);
    return inv_sqrt_2pi / std * b;
}

double ff(float x) {
    x = x / 100.0;
    return double(norm_pdf(0.0, 1.0, x) + norm_pdf(3.0, 1.0, x) + norm_pdf(6.0, 1.0, x));
}

__device__ float ff_gpu(float x) {
    x = x / 100.0;
    float a1 = norm_pdf_gpu(0.0, 1.0, x);
    float a2 = norm_pdf_gpu(3.0, 1.0, x);
    float a3 = norm_pdf_gpu(6.0, 1.0, x);;
    float result = a1 + a2 + a3;
    return result;
}

__device__ float(*ff_gpu_stat)(float) = ff_gpu;


std::tuple<float, double, double> _gen_candidate(float xt, float std, double f_xt, double (*f)(float), std::mt19937 gen) {
    std::normal_distribution<> g(xt, std);
    float xc = g(gen);
    double f_xc = f(xc);
    double a = f_xc / f_xt;
    return std::tuple<float, double, double>(xc, a, f_xc);
}


//__device__ std::tuple<float, double, double> _gen_candidate_gpu(float xt, float std, double f_xt, double (*f)(float)) {
//
//    return std::tuple<float, double, double>(xc, a, f_xc);
//}

void _generate_loop(float* x, float xt, float std, int start_range, int end_range, double (*f)(float), std::mt19937 gen) {
    double f_xt = f(xt);
    std::tuple<float, double, double> res;
    std::uniform_real<double> uni(0.0, 1.0);
    for (int t = start_range; t < end_range; t++) {
        res = _gen_candidate(xt, std, f_xt, f, gen);
        float u = uni(gen);
        if (u <= std::get<1>(res)) {
            xt = std::get<0>(res);
            f_xt = std::get<2>(res);
        }
        x[t] = xt;
    }
}


std::tuple<float, float> _burn_loop( float xt, float std, int burn_N, double (*f)(float), std::mt19937 gen) {
    double f_xt = f(xt);
    std::tuple<float, double, double> res;
    std::uniform_real<double> uni(0, 1);
    const float target = 0.3;
    int accepted = 0;
    for (int t = 0; t < burn_N; t++) {
        res = _gen_candidate(xt, std, f_xt, f, gen);
        float u = uni(gen);
        if (u <= std::get<1>(res)) {
            xt = std::get<0>(res);
            f_xt = std::get<2>(res);
            accepted += 1;
        }
        float temp = accepted /( t + 1.0);
        float reg = 1/std*1000.0 * (temp - target);
        if (!(reg < 0 && abs(reg) > std)) {
            std += reg;
        }
    }
    return std::tuple<float, float>(xt, std);
}

float* mh_v1(float x0, int N, int burn_N, double (*f)(float), int seed) {
    float* x = (float*) malloc(sizeof(float) * N);
    float xt = x0;
    float std = 1.0;

    std::mt19937 gen(seed);

    std::tuple<float, float> res;
    res = _burn_loop( xt, std, burn_N, f, gen);
    printf("std = %f", std::get<1>(res));
    _generate_loop(x, std::get<0>(res), std::get<1>(res), 0, N, f, gen);
    
    return x;
}

__global__ void _generate_loop_gpu(curandStateMtgp32* state, float* x, float xt, float std, float (*f)(float)) {
    int i = PER_JOB * blockIdx.x * BLOCK_SIZE + threadIdx.x * PER_JOB;
    int sstart_range = i ;
    int eend_range= i + PER_JOB;

    if (eend_range > NN) { return; }
  
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

cudaError_t mh_v2(float*x,float x0, int N, int burn_N, float (*f)(float), int seed) {

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }   

    float* dev_x;

    cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    float xt = x0;
    float std = 1.0;

    std::mt19937 gen(seed);

    std::tuple<float, float> res;
    res = _burn_loop( xt, std, burn_N, ff, gen);
    printf("std = %f", std::get<1>(res));

    curandStateMtgp32* devMTGPStates;
    mtgp32_kernel_params* devKernelParams;

    cudaMalloc((void**)&devMTGPStates, 64 * sizeof(curandStateMtgp32));

    cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));

    /* Reformat from predefined parameter sets to kernel format, */
    /* and copy kernel parameters to device memory               */
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);

    /* Initialize one state per thread block */
    curandMakeMTGP32KernelState(devMTGPStates,mtgp32dc_params_fast_11213, devKernelParams, (N / BLOCK_SIZE / PER_JOB), 1234);

    float (*fff)(float);
    cudaMemcpyFromSymbol(&fff, ff_gpu_stat, sizeof(f)); //tu nie mogę podać f po prostu tylko muszę ff_gpu_stat

    _generate_loop_gpu<<<(N/BLOCK_SIZE/PER_JOB),BLOCK_SIZE>>>(devMTGPStates,dev_x, std::get<0>(res), std::get<1>(res), fff);


    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(x, dev_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_x);

    return cudaStatus;
}








extern "C" int cuda_main(int argc, char** argv)
{


    int N = NN;
    float* xx = (float*)malloc(sizeof(float) * N);

    typedef std::chrono::high_resolution_clock Clock;
    Clock::time_point t0 = Clock::now();

   

    cudaError_t cudaStatus = mh_v2(xx,0.0, NN, 100000, ff_gpu_stat, 42141241);




    Clock::time_point t1 = Clock::now();
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    printf("%d \n", d.count());





    FILE* f = fopen("result.csv", "w");
    for (int i = 0; i < N; i++) {
        fprintf(f, "%f\n", xx[i]);
    }
    fclose(f);


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

