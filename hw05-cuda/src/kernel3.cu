#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

// Kernel 3 的重點是「Thread Coarsening (執行緒粗粒化)」
// 簡單說：原本一人做一點，現在一人做多點 (這裡是一人做 4 點)
// 好處：減少了總執行緒數量，減少了 GPU 排程的負擔，且每個執行緒做更多事可以隱藏一些延遲
#define GROUP_SIZE 4

__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, int res_x, int res_y, int max_iterations, size_t pitch) {
    // 1. 計算起始座標
    // 注意：現在每個執行緒負責 GROUP_SIZE 個像素
    // 所以全域 x 座標要乘以 GROUP_SIZE
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisY >= res_y)
        return;

    // 2. 迴圈處理這 4 個像素
    // 每個執行緒會連續處理 (thisX, thisY), (thisX+1, thisY), ..., (thisX+3, thisY)
    for (int k = 0; k < GROUP_SIZE; k++) {
        int currentX = thisX + k;
        
        // 邊界檢查：因為是粗粒化，最後一個 Block 的最後幾個執行緒可能會超出範圍
        if (currentX >= res_x) break;

        float c_re = lower_x + currentX * step_x;
        float c_im = lower_y + thisY * step_y;

        float z_re = c_re, z_im = c_im;
        int i;
        for (i = 0; i < max_iterations; ++i) {
            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }

        // 寫入結果
        // 一樣要用 pitch 來計算正確的記憶體位址
        int *row = (int *)((char *)img + thisY * pitch);
        row[currentX] = i;
    }
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int *img, int res_x, int res_y, int max_iterations) {
    float step_x = (upper_x - lower_x) / res_x;
    float step_y = (upper_y - lower_y) / res_y;

    int *d_img;
    size_t pitch;
    
    cudaMallocPitch((void **)&d_img, &pitch, res_x * sizeof(int), res_y);

    int *h_img;
    cudaHostAlloc((void **)&h_img, res_x * res_y * sizeof(int), cudaHostAllocDefault);

    dim3 blockSize(16, 16);
    
    // Grid 設定改變了！
    // X 方向的 Grid 數量只需要原本的 1/GROUP_SIZE
    // 因為每個執行緒現在處理 GROUP_SIZE 個像素
    dim3 gridSize((res_x + blockSize.x * GROUP_SIZE - 1) / (blockSize.x * GROUP_SIZE), (res_y + blockSize.y - 1) / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations, pitch);

    cudaMemcpy2D(h_img, res_x * sizeof(int), d_img, pitch, res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost);

    memcpy(img, h_img, res_x * res_y * sizeof(int));

    cudaFree(d_img);
    cudaFreeHost(h_img);
}
