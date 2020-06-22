//
//  vector_add.cpp
//
//  Creado por Guadalupe Flores 22/06/20.
//
//  Suma de dos vectores con 100 millones de elementos, la ejecución del mismo se realizará en CPU
//  como variable de control entre programacion con Serial y Paralela con CUDA.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <ctime> 

#define N 100000000
#define MAX_ERR 1e-6

void vector_add(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float* a, * b, * out;
    unsigned t0, t1;
    
    t0 = clock();
    // Asigno memoria
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Inicializo el array
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Llamo a la función
    vector_add(out, a, b, N);

    // Verificación
    for (int i = 0; i < N; i++) {
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("out[0] = %f\n", out[0]);
    printf("Finalizado\n");
    t1 = clock();

    double time = ((double(t1 - t0) / CLK_TCK));
    printf("Tiempo de ejecucion: = %d \n", time);
    


    // Libero la memoria
    free(a);
    free(b);
    free(out);
}