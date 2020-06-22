//
//  vector_add.cu
//
//  Creado por Guadalupe Flores 22/06/20.
//
//  Suma de vectores con 100 millones de registros utilizando 1 thread, 1 block de CUDA.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100000000 //100.000.000
#define MAX_ERR 1e-6

__global__ void vector_add(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float* a, * b, * out;
    float* d_a, * d_b, * d_out;

    // Asigno memoria host
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Inicializo los array host
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Asigno memoria en la GPU
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfiero la memoria
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Ejecuto el kernel 
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    // Transfiero la memoria de vuelta para traer el resultado.
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verificamos 
    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED CUDA\n");

    // Liberamos memoria del GPU    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Liberamos la memoria del host. 
    free(a);
    free(b);
    free(out);
}
