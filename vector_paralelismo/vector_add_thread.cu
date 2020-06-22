//
//  vector_add_thread.cu
//
//  Creado por Guadalupe Flores 22/06/20.
//
//  Suma de vectores con 100 millones de registros utilizando 256 threads y 1 block,
//  esto demuestra como el paralelismo con threads mejora la velocidad de procesamiento
//  de los vectores.
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

    // contiene el indice del thread en el bloque
    // cada hilo se va a encargar de un dato distinto
    // asi vamos a asignar distintas direcciones de memoria a distintos hilos
    int index = threadIdx.x; 
    int stride = blockDim.x; // 256

    // esto va a hacer que en cada iteración se procesen 256 posiciones en simultaneo.
    for (int i = index; i < n; i += stride) {
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

    // Transfiero la memoria desde el host a GPU
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Ejecuto el kernel
    // Lo que indica que el kernel se inicia con un 1 bloques de hilos. 
    // Cada bloque tiene 256 hilos paralelos.
    // Funcion que se ejecuta al modo SPMD 
    vector_add << <1, 256 >> > (d_out, d_a, d_b, N);

    // Transfer data back to host memory
    // Transferimos los datos de vuelta a la memoria del host. 
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
