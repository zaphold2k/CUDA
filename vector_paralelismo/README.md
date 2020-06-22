# Ejemplo de suma de dos vectores con CUDA

Requisitos:
- CUDA Toolkit 4.0+

Este codigo ejemplo se compone de 4 archivos independientes, que demuestran las diferentes implementaciones de "kernel launch" y los beneficios que otorgan a la programación paralela.

### vector_add.cpp
Suma de dos vectores con 100 millones de elementos, la ejecución del mismo se realizará en CPU como variable de control entre programacion Serial y Paralela con CUDA.

### vector_add.cu
La misma suma de vectores que el anterior, pero utilizando 1 thread, 1 block de CUDA.

### vector_add_thread.cu
Suma de los vectores, pero utilizando 256 threads y 1 block, esto demuestra como el paralelismo con threads mejora la velocidad de procesamiento de los vectores.

### vector_add_grid.cu
Por último, la suma de los vectores se realiza con bloques de 256 threads (total 390.626 blocks), aumentando la capacidad de procesamiento, implementando al maximo el paralelismo y por consecuente el tiempo de ejecución.

Ejemplos creados siguiendo el [Workshop de CUDA](https://cuda-tutorial.readthedocs.io/en/latest/). 

*El código se encuentra documentado con comentarios a fines didácticos.*
