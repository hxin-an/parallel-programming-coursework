#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"

// Kernel 4: 競賽極速版 (Final Version)
// 
// 優化策略總結：
// 1. ILP (Instruction Level Parallelism) = 2:
//    每個執行緒同時處理 2 個像素 (GROUP_SIZE = 2)。
//    這能讓編譯器更有效地排程指令，隱藏運算延遲，同時避免過高的暫存器壓力 (Register Pressure)。
//
// 2. Loop Unrolling (Factor 32):
//    主迴圈手動展開 32 次。
//    - 減少了大量的分支指令 (Branching) 和迴圈計數器更新。
//    - 讓指令管線 (Pipeline) 更順暢。
//
// 3. Periodicity Checking (週期性檢查) - 針對 View 2 優化:
//    - 許多在集合內的點會進入循環 (Cycle)，永遠不會發散。
//    - 我們每 32 個迭代檢查一次 z 值是否重複。
//    - 如果發現重複 (z == history)，直接判定為收斂，提早結束。
//    - 這對 Deep Zoom (View 2) 是關鍵優化，能省下數萬次無效迭代。
//
// 4. Cardioid / Bulb Check (幾何檢查) - 針對 View 1 優化:
//    - 快速排除 Mandelbrot 集合中最大的愛心 (Cardioid) 和圓形 (Bulb) 區域。
//    - 這些區域內的點永遠不會發散。
//    - 加上 0.95 的安全係數 (Safety Factor) 以避免邊界浮點數誤差導致的正確性問題。
//
// 5. __launch_bounds__(128):
//    - 明確告知編譯器 Block Size 為 128，協助優化暫存器分配。

#define GROUP_SIZE 2

// 定義單步更新的 Macro
// k: 當前是第幾個 unroll step (0~31)
#define STEP(k) \
    if (act0) { \
        float sq_re = z_re0 * z_re0; \
        float sq_im = z_im0 * z_im0; \
        if (sq_re + sq_im > 4.f) { \
            act0 = false; \
            iter0 = i + k; \
        } else { \
            float new_re = sq_re - sq_im + c_re0; \
            float new_im = 2.f * z_re0 * z_im0 + c_im; \
            z_re0 = new_re; \
            z_im0 = new_im; \
        } \
    } \
    if (act1) { \
        float sq_re = z_re1 * z_re1; \
        float sq_im = z_im1 * z_im1; \
        if (sq_re + sq_im > 4.f) { \
            act1 = false; \
            iter1 = i + k; \
        } else { \
            float new_re = sq_re - sq_im + c_re1; \
            float new_im = 2.f * z_re1 * z_im1 + c_im; \
            z_re1 = new_re; \
            z_im1 = new_im; \
        } \
    }

__global__ void __launch_bounds__(128) mandel_kernel(float lower_x, float lower_y, float step_x, float step_y, int *img, int res_x, int res_y, int max_iterations, size_t pitch) {
    // 計算起始 x 座標 (每個執行緒負責 2 個點)
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisY >= res_y) return;

    // 暫存器變數 (Registers)
    float c_re0, c_re1;
    float z_re0, z_re1;
    float z_im0, z_im1;
    int iter0 = 0, iter1 = 0;
    
    // Periodicity Check 變數
    float h_re0, h_re1;
    float h_im0, h_im1;
    int period0 = 1, period1 = 1;
    int check0 = 0, check1 = 0;

    float c_im = lower_y + thisY * step_y;
    float y2 = c_im * c_im;

    bool act0 = (thisX < res_x);
    bool act1 = (thisX + 1 < res_x);

    // 1. Cardioid / Bulb Check (幾何檢查)
    // 預先排除必定收斂的區域
    #pragma unroll
    for(int k=0; k<GROUP_SIZE; k++) {
        int currentX = thisX + k;
        if (currentX < res_x) {
            float x = lower_x + currentX * step_x;
            float q = (x - 0.25f) * (x - 0.25f) + y2;
            
            // Cardioid Check (含 0.95 安全係數)
            if (q * (q + (x - 0.25f)) < 0.25f * y2 * 0.95f) {
                if(k==0) { iter0 = max_iterations; act0 = false; }
                if(k==1) { iter1 = max_iterations; act1 = false; }
            } 
            // Period-2 Bulb Check (含 0.95 安全係數)
            else if ((x + 1.f) * (x + 1.f) + y2 < 0.0625f * 0.95f) {
                if(k==0) { iter0 = max_iterations; act0 = false; }
                if(k==1) { iter1 = max_iterations; act1 = false; }
            } else {
                // 必須計算的點，初始化變數
                if(k==0) { c_re0 = x; z_re0 = x; z_im0 = c_im; h_re0 = x; h_im0 = c_im; }
                if(k==1) { c_re1 = x; z_re1 = x; z_im1 = c_im; h_re1 = x; h_im1 = c_im; }
            }
        } else {
            if(k==0) act0 = false;
            if(k==1) act1 = false;
        }
    }

    int i = 0;
    // 2. 主迴圈：每次處理 32 個迭代 (Loop Unrolling)
    for (; i + 32 <= max_iterations; i += 32) {
        if (!act0 && !act1) break;

        // 展開 32 次計算
        STEP(0); STEP(1); STEP(2); STEP(3);
        STEP(4); STEP(5); STEP(6); STEP(7);
        STEP(8); STEP(9); STEP(10); STEP(11);
        STEP(12); STEP(13); STEP(14); STEP(15);
        STEP(16); STEP(17); STEP(18); STEP(19);
        STEP(20); STEP(21); STEP(22); STEP(23);
        STEP(24); STEP(25); STEP(26); STEP(27);
        STEP(28); STEP(29); STEP(30); STEP(31);

        // 3. Periodicity Check (每 32 次執行一次)
        // 檢查是否進入循環，如果是，則提早結束
        if (act0) {
            if (z_re0 == h_re0 && z_im0 == h_im0) {
                act0 = false;
                iter0 = max_iterations;
            } else {
                check0 += 32;
                if (check0 >= period0) {
                    check0 = 0;
                    period0 <<= 1; // 週期加倍 (Brent's Algorithm 概念)
                    h_re0 = z_re0;
                    h_im0 = z_im0;
                }
            }
        }
        if (act1) {
            if (z_re1 == h_re1 && z_im1 == h_im1) {
                act1 = false;
                iter1 = max_iterations;
            } else {
                check1 += 32;
                if (check1 >= period1) {
                    check1 = 0;
                    period1 <<= 1;
                    h_re1 = z_re1;
                    h_im1 = z_im1;
                }
            }
        }
    }

    // 處理剩下的迭代 (Tail)
    for (; i < max_iterations; ++i) {
        if (!act0 && !act1) break;
        STEP(0); // 這裡 k=0 是相對位移，實際上 iter = i + 0
    }
    
    // 如果跑完還沒發散，設為 max
    if (act0) iter0 = max_iterations;
    if (act1) iter1 = max_iterations;

    // 寫入結果
    int *row = (int *)((char *)img + thisY * pitch);
    if (thisX < res_x) row[thisX] = iter0;
    if (thisX + 1 < res_x) row[thisX + 1] = iter1;
}

void host_fe(float upper_x, float upper_y, float lower_x, float lower_y, int *img, int res_x, int res_y, int max_iterations) {
    float step_x = (upper_x - lower_x) / res_x;
    float step_y = (upper_y - lower_y) / res_y;

    int *d_img;
    size_t pitch;
    cudaMallocPitch((void **)&d_img, &pitch, res_x * sizeof(int), res_y);

    int *h_img;
    cudaHostAlloc((void **)&h_img, res_x * res_y * sizeof(int), cudaHostAllocDefault);

    // Block 大小：32x4
    dim3 blockSize(32, 4);
    // Grid 大小：X 方向除以 GROUP_SIZE (2)
    dim3 gridSize((res_x + blockSize.x * GROUP_SIZE - 1) / (blockSize.x * GROUP_SIZE), (res_y + blockSize.y - 1) / blockSize.y);

    mandel_kernel<<<gridSize, blockSize>>>(lower_x, lower_y, step_x, step_y, d_img, res_x, res_y, max_iterations, pitch);

    cudaMemcpy2D(h_img, res_x * sizeof(int), d_img, pitch, res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost);
    memcpy(img, h_img, res_x * res_y * sizeof(int));

    cudaFree(d_img);
    cudaFreeHost(h_img);
}
