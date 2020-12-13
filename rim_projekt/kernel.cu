#define _USE_MATH_DEFINES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <chrono>

#include <stdio.h>
#include <random>
#include <tuple>
#include <math.h>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



double norm_pdf(double mean, float std, float x) {
    const double a = 1 / sqrt(2 * M_PI);
    return a / std * exp(-0.5 * pow(((x - mean) / std), 2));
}

double ff(float x) {
    x = x / 100.0;
    return norm_pdf(0.0, 1.0, x) + norm_pdf(3.0, 1.0, x) + norm_pdf(6.0, 1.0, x);
}

std::tuple<float, double, double> _gen_candidate(float xt, float std, double f_xt, double (*f)(float), std::mt19937 gen) {
    std::normal_distribution<> g(xt, std);
    float xc = g(gen);
    double f_xc = f(xc);
    double a = f_xc / f_xt;
    return std::tuple<float, double, double>(xc, a, f_xc);
}

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

std::tuple<float, float> _burn_loop(float* x, float xt, float std, int burn_N, double (*f)(float), std::mt19937 gen) {
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
    res = _burn_loop(x, xt, std, burn_N, f, gen);
    printf("std = %f", std::get<1>(res));
    _generate_loop(x, std::get<0>(res), std::get<1>(res), 0, N, f, gen);
    
    return x;
}






int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };


    int N = 100000;

    typedef std::chrono::high_resolution_clock Clock;
    Clock::time_point t0 = Clock::now();
    float* x = mh_v1(0.0, N, 100000, ff, 42141241);
    Clock::time_point t1 = Clock::now();
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    printf("%d \n", d.count());

    FILE* f = fopen("result.csv", "w");
    for (int i = 0; i < N; i++) {
        fprintf(f, "%f\n", x[i]);
    }
    fclose(f);

    //FILE* f = fopen("result.csv", "w");
    //for (int i = -500; i < 1200; i++) {
    //    fprintf(f, "%f\n", ff(1.0*i));
    //}
    //fclose(f);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
