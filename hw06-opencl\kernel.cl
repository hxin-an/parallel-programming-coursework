__kernel void convolution(int filter_width,
                          __constant float *filter,
                          int image_height,
                          int image_width,
                          __global float *input_image,
                          __global float *output_image)
{
    // 1. 取得當前 Thread 的座標 (x, y)
    // get_global_id(0) 對應到 image_width (x)
    // get_global_id(1) 對應到 image_height (y)
    int x = get_global_id(0);
    int y = get_global_id(1);

    // 2. 邊界檢查
    // 如果座標超出影像範圍，則不進行計算
    if (x >= image_width || y >= image_height)
        return;

    int halffilter_size = filter_width / 2;
    float sum = 0.0f;
    int k, l;

    // 3. 卷積運算
    // 遍歷 Filter 覆蓋的區域
    for (k = -halffilter_size; k <= halffilter_size; k++)
    {
        for (l = -halffilter_size; l <= halffilter_size; l++)
        {
            // 計算鄰居像素的座標
            int r = y + k; // Row
            int c = x + l; // Column

            // Zero-padding: 檢查鄰居像素是否在影像範圍內
            if (r >= 0 && r < image_height && c >= 0 && c < image_width)
            {
                // 讀取像素值並乘上對應的 Filter 權重
                // input_image index: r * width + c
                // filter index: (k + halffilter_size) * filter_width + (l + halffilter_size)
                sum += input_image[r * image_width + c] *
                       filter[(k + halffilter_size) * filter_width + (l + halffilter_size)];
            }
        }
    }

    // 4. 寫入結果
    output_image[y * image_width + x] = sum;
}
