#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;
    int filter_size = filter_width * filter_width;
    size_t image_data_size = image_height * image_width * sizeof(float);
    size_t filter_data_size = filter_size * sizeof(float);

    // 1. 建立 Command Queue (命令隊列)
    // 用來傳送指令給 GPU。
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");

    // 2. 建立 Memory Buffers (記憶體緩衝區)
    // 在 GPU 上配置記憶體空間。
    // input_buffer: 唯讀，存放輸入影像
    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_data_size, NULL, &status);
    CHECK(status, "clCreateBuffer input");
    
    // filter_buffer: 唯讀，存放濾鏡權重
    cl_mem filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filter_data_size, NULL, &status);
    CHECK(status, "clCreateBuffer filter");

    // output_buffer: 唯寫，存放輸出結果
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_data_size, NULL, &status);
    CHECK(status, "clCreateBuffer output");

    // 3. 寫入資料到 Device (GPU)
    // 將 Host 端的資料複製到 GPU 的 Buffer 中。
    status = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, image_data_size, input_image, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer input");

    status = clEnqueueWriteBuffer(queue, filter_buffer, CL_TRUE, 0, filter_data_size, filter, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer filter");

    // 4. 建立 Kernel (核心程式)
    // 從編譯好的 Program 中取出名為 "convolution" 的 Kernel。
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // 5. 設定 Kernel 參數
    // 對應到 kernel.cl 中的參數順序。
    status = clSetKernelArg(kernel, 0, sizeof(int), &filter_width);
    CHECK(status, "clSetKernelArg 0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &filter_buffer);
    CHECK(status, "clSetKernelArg 1");
    status = clSetKernelArg(kernel, 2, sizeof(int), &image_height);
    CHECK(status, "clSetKernelArg 2");
    status = clSetKernelArg(kernel, 3, sizeof(int), &image_width);
    CHECK(status, "clSetKernelArg 3");
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_buffer);
    CHECK(status, "clSetKernelArg 4");
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_buffer);
    CHECK(status, "clSetKernelArg 5");

    // 6. 執行 Kernel
    // 設定全域工作大小 (Global Work Size)，對應到影像的寬高。
    size_t global_work_size[2] = { (size_t)image_width, (size_t)image_height };
    // Local Work Size 先設為 NULL，讓 OpenCL 自動決定。
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // 7. 讀回結果
    // 將 GPU 計算完的結果複製回 Host 端。
    status = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, image_data_size, output_image, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer output");

    // 8. 釋放資源
    clReleaseKernel(kernel);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(filter_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(queue);
}
