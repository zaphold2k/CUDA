//
//  vector_add_grid.cu
//
//  Creado por Guadalupe Flores 22/06/20.
//
//  Suma de vectores con 100 millones de registros realizada con bloques de 256 threads
//  (total 390.626 blocks), aumentando la capacidad de procesamiento, implementando al
//  máximo el paralelismo y por consecuente el tiempo de ejecución.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100000000
#define MAX_ERR 1e-6

__global__ void vector_add(float* out, float* a, float* b, int n) {

    // Para asignar un thread a un elemento específico, necesitamos conocer un índice único para cada thread.
    // Tal índice se puede calcular de la siguiente manera
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // donde: 
    // blockIdx.x contiene el índice del bloque con en la cuadrícula
    // gridDim.x contiene el tamaño de la cuadrícula


    // Como todo el array queda cubierto remuevo el for y solo valido que tid no sea mayor que el tama�o 
    // del array que estamos sumando
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
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
    // Con 256 hilos por bloque de hilos, necesitamos al menos N / 256 bloques de hilos 
    // para tener un total de N hilos.
    int block_size = 256;
    int grid_size = ((N + block_size) / block_size); // 390.626

    printf("N = %d \n", N);
    printf("block_size = %d \n", block_size);
    printf("grid_size = %d \n", grid_size);

    vector_add << <grid_size, block_size >> > (d_out, d_a, d_b, N);
    

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a);
    free(b);
    free(out);
}
