//
//  imgproc.cu
//  
//
//  Created by Nathaniel Lewis on 3/8/12.
//  Copyright (c) 2012 E1FTW Games. All rights reserved.
//
//  @2020
//  Modificado y traducido al español para utilizarlo con fines didácticos.
//  Martin Casanovas
//  Guadalupe Flores

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h> 
#include <stdlib.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

// Constante de Memoria de la GPU para mantener los kernels
__constant__ float convolutionKernelStore[256];

 /**
 * Función de convolución para cuda. Se espera que el destino tenga el mismo ancho / alto que el origen, pero habrá un borde
 * de píxeles de piso (kWidth / 2) a izquierda y derecha y píxeles de piso (kHeight / 2) arriba y abajo
 *
 * @param source      Puntero de memoria fijada en host de imagen de origen
 * @param width       Ancho de la imagen de origen
 * @param height      Altura de la imagen de origen
 * @param paddingX fuente de imagen a lo largo de x
 * @param paddingY fuente de imagen a lo largo de y
 * @param kOffset offset en memoria constante del almac�n del kernel
 * @param kWidth kernel width
 * @param kHeight altura del kernel
 * @param destino Imagen de destino host anclado puntero de memoria
 */
__global__ void convolve(unsigned char* source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char* destination)
{
    // Calculamos la ubicación de nuestro píxel
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float sum = 0.0;
    int   pWidth = kWidth / 2;
    int   pHeight = kHeight / 2;

    // Solo ejecuta los pixeles validos
    if (x >= pWidth + paddingX &&
        y >= pHeight + paddingY &&
        x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
        y < (blockDim.y * gridDim.y) - pHeight - paddingY)
    {
        for (int j = -pHeight; j <= pHeight; j++)
        {
            for (int i = -pWidth; i <= pWidth; i++)
            {
                int ki = (i + pWidth);
                int kj = (j + pHeight);
                float w = convolutionKernelStore[(kj * kWidth) + ki + kOffset];


                sum += w * float(source[((y + j) * width) + (x + i)]);
            }
        }
    }

    // Promedio de la suma
    destination[(y * width) + x] = (unsigned char)sum;
}

// convierte el teorema de Pitágoras a lo largo de un vector en la GPU
__global__ void pythagoras(unsigned char* a, unsigned char* b, unsigned char* c)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float af = float(a[idx]);
    float bf = float(b[idx]);

    c[idx] = (unsigned char)sqrtf(af * af + bf * bf);
}

// crear un búfer de imagen. devolver host ptr, pasar el puntero del dispositivo a través del puntero al puntero (153)
unsigned char* createImageBuffer(unsigned int bytes, unsigned char** devicePtr)
{
    unsigned char* ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(devicePtr, ptr, 0);
    return ptr;
}


int main(int argc, char** argv)
{
    // Abrimos la cámara
    cv::VideoCapture camera(0);
    cv::Mat          frame;
    if (!camera.isOpened())
        return -1;

    // Creamos la ventana de la captura
    cv::namedWindow("Source");
    cv::namedWindow("Greyscale");
    cv::namedWindow("Blurred");
    cv::namedWindow("Sobel");
    
    // Creamos los timers de CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Preparamos el kernel gaussiano
    const float gaussianKernel5x5[25] =
    {
        2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
        4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
        5.f / 159.f, 12.f / 159.f, 15.f / 159.f, 12.f / 159.f, 5.f / 159.f,
        4.f / 159.f,  9.f / 159.f, 12.f / 159.f,  9.f / 159.f, 4.f / 159.f,
        2.f / 159.f,  4.f / 159.f,  5.f / 159.f,  4.f / 159.f, 2.f / 159.f,
    };

    cudaMemcpyToSymbol(convolutionKernelStore, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0); 
    // convolutionKernelStore: Constante de Memoria de la GPU para mantener los kernels
    const ssize_t gaussianKernel5x5Offset = 0;

    // Preparamos los kernels para el filtro de suavizar 
    const float sobelGradientX[9] =
    {
        -1.f, 0.f, 1.f,
        -2.f, 0.f, 2.f,
        -1.f, 0.f, 1.f,
    };
    const float sobelGradientY[9] =
    {
        1.f, 2.f, 1.f,
        0.f, 0.f, 0.f,
        -1.f, -2.f, -1.f,
    };
    cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientX, sizeof(sobelGradientX), sizeof(gaussianKernel5x5));
    cudaMemcpyToSymbol(convolutionKernelStore, sobelGradientY, sizeof(sobelGradientY), sizeof(gaussianKernel5x5) + sizeof(sobelGradientX));
    const ssize_t sobelGradientXOffset = sizeof(gaussianKernel5x5) / sizeof(float);
    const ssize_t sobelGradientYOffset = sizeof(sobelGradientX) / sizeof(float) + sobelGradientXOffset;

    // Cree imágenes compartidas de CPU / GPU: una para la inicial y otra para el resultado
    camera >> frame;
    unsigned char* sourceDataDevice, * blurredDataDevice, * edgesDataDevice;
    // Crea tres matrices para luego poder operar y asigna memoria para operar en la GPU
    cv::Mat source(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &sourceDataDevice));
    cv::Mat blurred(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &blurredDataDevice));
    cv::Mat edges(frame.size(), CV_8U, createImageBuffer(frame.size().width * frame.size().height, &edgesDataDevice));
    
    // Creamos dos imágenes temporiarias para mantener los gradientes del filtro suavizado
    unsigned char* deviceGradientX, * deviceGradientY;
    cudaMalloc(&deviceGradientX, frame.size().width * frame.size().height);
    cudaMalloc(&deviceGradientY, frame.size().width * frame.size().height);

    // Loop mientras se captura la imágen.
    while (1)
    {
        // Capturamos la imagen y almacenenamos una conversión gris a la GPU
        camera >> frame;
        cv::cvtColor(frame, source, cv::COLOR_BGR2GRAY);

        // Registramos el tiempo que lleva procesar
        cudaEventRecord(start);
        {
            // inicializacion de parametros para el kernel de convolve
            dim3 cblocks(frame.size().width / 16, frame.size().height / 16);
            dim3 cthreads(16, 16);

            // inicializacion de parametros para el algoritmo de pitagoras
            dim3 pblocks(frame.size().width * frame.size().height / 256);
            dim3 pthreads(256, 1);

            // Realizamos el desenfoque gaussiano
            convolve << < cblocks, cthreads >> > (sourceDataDevice, frame.size().width, frame.size().height, 0, 0, gaussianKernel5x5Offset, 5, 5, blurredDataDevice);
            // Realizar las convoluciones de gradiente sobel (el padding x & y ahora es 2 porque hay un borde de 2 alrededor de una imagen gaussiana filtrada de 5x5)            
            convolve << < cblocks, cthreads >> > (blurredDataDevice, frame.size().width, frame.size().height, 2, 2, sobelGradientXOffset, 3, 3, deviceGradientX);
            convolve << < cblocks, cthreads >> > (blurredDataDevice, frame.size().width, frame.size().height, 2, 2, sobelGradientYOffset, 3, 3, deviceGradientY);
          
            pythagoras << < pblocks, pthreads >> > (deviceGradientX, deviceGradientY, edgesDataDevice);

            // Se sincronizan los hilos para asegurarse que se procesa la imagen completa. 
            cudaThreadSynchronize();
        }
        cudaEventRecord(stop);

        // Mostrarmos el tiempo transcurrido.
        float ms = 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Elapsed GPU time: " << ms << " milliseconds" << std::endl;

        // Mostramos los resultados. 
        cv::imshow("Source", frame);
        cv::imshow("Greyscale", source);
        cv::imshow("Blurred", blurred);
        cv::imshow("Sobel", edges);

        if (cv::waitKey(1) == 27) break;
    }

    // Salimos
    cudaFreeHost(source.data);
    cudaFreeHost(blurred.data);
    cudaFreeHost(edges.data);
    cudaFree(deviceGradientX);
    cudaFree(deviceGradientY);

    return 0;
}
