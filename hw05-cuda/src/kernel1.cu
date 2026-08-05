#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

// __global__ 表示這個函式是在 GPU 上執行，但是由 CPU 呼叫
// 這是我們的 "Kernel" (核心函式)，每個 GPU 執行緒都會執行這段程式碼
__global__ void mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, int res_x, int res_y, int max_iterations) {
    // 1. 計算當前執行緒負責的像素座標 (x, y)
    // blockIdx.x: 目前是第幾個 Block (區塊)
    // blockDim.x: 每個 Block 有多少個 Thread (執行緒)
    // threadIdx.x: 目前是 Block 內的第幾個 Thread
    // 公式：全域 ID = (區塊 ID * 區塊大小) + 區塊內 ID
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. 邊界檢查
    // 因為我們的 Grid (網格) 大小通常會補齊到 Block 的倍數，可能會比圖片實際大小還大
    // 所以如果算出來的座標超過圖片範圍，就直接離開，不做事
    if (thisX >= res_x || thisY >= res_y)
        return;

    // 3. 座標映射：將像素座標 (thisX, thisY) 轉換到複數平面座標 (c_re, c_im)
    // c_re = 起始 x + (第幾格 x * 每格寬度)
    float c_re = lower_x + thisX * step_x;
    float c_im = lower_y + thisY * step_y;

    // 4. 曼德博集合 (Mandelbrot Set) 計算核心
    // 公式：z_{n+1} = z_n^2 + c
    // 初始值 z_0 = c (這裡作業的邏輯是從 c 開始，或者 z=0 開始迭代第一次也是 c)
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < max_iterations; ++i) {
        // 判斷是否發散：如果模長的平方 (real^2 + imag^2) > 4，表示發散
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        // 計算下一次迭代
        // z^2 = (a+bi)^2 = a^2 - b^2 + 2abi
        // 新的實部 = 舊實部平方 - 舊虛部平方 + c的實部
        float new_re = z_re * z_re - z_im * z_im;
        // 新的虛部 = 2 * 舊實部 * 舊虛部 + c的虛部
        float new_im = 2.f * z_re * z_im;
        
        // 更新 z
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    // 5. 將計算結果 (迭代次數) 寫入全域記憶體
    // 計算一維陣列的索引值：行號 * 寬度 + 列號
    int index = thisY * res_x + thisX;
    img[index] = i;
}

// Host 端 (CPU) 的前端函式，負責準備記憶體和呼叫 GPU
void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int *img, int res_x, int res_y, int max_iterations) {
    // 計算每個像素代表的複數平面距離 (步長)
    float step_x = (upper_x - lower_x) / res_x;
    float step_y = (upper_y - lower_y) / res_y;

    // d_img: Device (GPU) 端的記憶體指標
    int *d_img;
    // 計算整張圖需要的 byte 大小
    int size = res_x * res_y * sizeof(int);

    // 1. 配置 GPU 記憶體
    // cudaMalloc(指標的位址, 大小)
    cudaMalloc((void **)&d_img, size);

    // 2. 配置 Host (CPU) 暫存記憶體
    // 作業要求 Kernel 1 必須使用 new 來配置
    // 我們不能直接用傳進來的 img，因為它可能不是我們能控制的 (雖然在這裡通常沒差，但為了符合規定)
    int *h_img = new int[res_x * res_y];

    // 3. 設定執行組態 (Grid 和 Block)
    // Block 大小：我們設定每個 Block 有 16x16 = 256 個執行緒
    dim3 blockSize(16, 16);
    
    // Grid 大小：計算需要多少個 Block 才能蓋滿整張圖
    // 公式：(圖片大小 + Block大小 - 1) / Block大小
    // 這裡的 + blockSize.x - 1 是為了無條件進位 (Ceiling)
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, (res_y + blockSize.y - 1) / blockSize.y);

    // 4. 啟動 Kernel (GPU 開始工作)
    // 語法：Kernel名稱<<<Grid數量, Block大小>>>(參數...)
    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations);

    // 5. 將結果從 GPU 搬回 CPU
    // cudaMemcpy(目的, 來源, 大小, 方向)
    // 方向：cudaMemcpyDeviceToHost (從裝置到主機)
    cudaMemcpy(h_img, d_img, size, cudaMemcpyDeviceToHost);

    // 6. 將結果複製到最終輸出的 buffer
    for (int i = 0; i < res_x * res_y; i++) {
        img[i] = h_img[i];
    }

    // 7. 釋放記憶體
    // 釋放 GPU 記憶體
    cudaFree(d_img);
    // 釋放 CPU 記憶體
    delete[] h_img;
}
