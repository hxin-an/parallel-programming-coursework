#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

// Kernel 2 的重點在於「記憶體優化」
// 邏輯跟 Kernel 1 一模一樣，但是讀寫記憶體的方式變了
__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, int res_x, int res_y, int max_iterations, size_t pitch) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= res_x || thisY >= res_y)
        return;

    float c_re = lower_x + thisX * step_x;
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

    // 重點來了！
    // 因為我們用了 cudaMallocPitch，記憶體每一行 (Row) 的長度可能不等於 res_x * sizeof(int)
    // 為了對齊，GPU 會在每一行後面補上一些 padding (空白)
    // 所以我們不能直接用 img[thisY * res_x + thisX] 來找位置
    
    // 正確做法：
    // 1. 先把 img 指標轉成 char* (byte 單位)，因為 pitch 是以 byte 為單位的
    // 2. 加上 thisY * pitch，跳到第 thisY 行的開頭 (這裡已經包含了 padding)
    // 3. 再轉回 int*，加上 thisX，找到這一行的第 thisX 個像素
    int *row = (int *)((char *)img + thisY * pitch);
    row[thisX] = i;
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int *img, int res_x, int res_y, int max_iterations) {
    float step_x = (upper_x - lower_x) / res_x;
    float step_y = (upper_y - lower_y) / res_y;

    int *d_img;
    // pitch: 用來存 GPU 實際配置的每一行寬度 (byte)
    size_t pitch;
    
    // 1. 配置 Device (GPU) 記憶體 - 使用 cudaMallocPitch
    // 為什麼？因為 GPU 喜歡讀取對齊的記憶體 (Coalesced Access)
    // 如果圖片寬度不是 32 或 64 的倍數，普通的 cudaMalloc 可能會讓讀取效率變差
    // cudaMallocPitch 會自動補齊寬度，讓每一行的開頭都在對齊的位址上
    cudaMallocPitch((void **)&d_img, &pitch, res_x * sizeof(int), res_y);

    // 2. 配置 Host (CPU) 記憶體 - 使用 cudaHostAlloc (Pinned Memory)
    // 為什麼？普通的 new 出來的記憶體是 Pageable 的，可能會被作業系統換到硬碟
    // Pinned Memory (鎖定記憶體) 保證一直待在 RAM 裡
    // 這樣 GPU 就可以透過 DMA (Direct Memory Access) 直接抓資料，不用 CPU 幫忙搬，速度快很多！
    int *h_img;
    cudaHostAlloc((void **)&h_img, res_x * res_y * sizeof(int), cudaHostAllocDefault);

    dim3 blockSize(16, 16);
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, (res_y + blockSize.y - 1) / blockSize.y);

    // 啟動 Kernel，記得把 pitch 傳進去
    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations, pitch);

    // 3. 搬移資料 - 使用 cudaMemcpy2D
    // 因為 GPU 上的資料有 padding (pitch)，而 CPU 上的資料是連續的 (width * sizeof(int))
    // 所以要用 2D 搬移函式，告訴它兩邊的寬度分別是多少
    // 參數：目的, 目的寬度(pitch), 來源, 來源寬度(pitch), 實際資料寬度, 高度, 方向
    cudaMemcpy2D(h_img, res_x * sizeof(int), d_img, pitch, res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost);

    // 複製到輸出 buffer
    memcpy(img, h_img, res_x * res_y * sizeof(int));

    // 4. 釋放記憶體
    cudaFree(d_img);
    // Pinned Memory 要用專用的釋放函式
    cudaFreeHost(h_img);
}
