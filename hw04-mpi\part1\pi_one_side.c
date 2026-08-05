#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // MPI_Win: RMA (Remote Memory Access) 窗口物件
    // 用於建立可被其他 processes 存取的共享記憶體區域
    MPI_Win win;

    // TODO: MPI init
    // 獲取 MPI 環境資訊
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // 總共有多少個 processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // 當前 process 的編號 (0 到 world_size-1)
    
    // 為每個 process 設定不同的隨機種子，確保產生不同的隨機數序列
    unsigned int seed = (world_rank + 1) * time(NULL);
    
    // 每個 process 負責的投擲次數 (簡單整數除法，會忽略餘數)
    long long int local_tosses = tosses / world_size;
    
    // 記錄本 process 計算出落在圓內的點數
    long long int number_in_circle = 0;
    
    // Monte Carlo 模擬：隨機產生點並判斷是否在單位圓內
    // 原理：在 [-1, 1] x [-1, 1] 的正方形內隨機取點
    //      若 x^2 + y^2 <= 1，則該點在單位圓內
    //      圓面積 / 正方形面積 = π/4
    for (long long int toss = 0; toss < local_tosses; toss++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;  // [-1, 1] 範圍的 x 座標
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;  // [-1, 1] 範圍的 y 座標
        double distance_squared = x * x + y * y;                  // 計算點到原點的距離平方
        if (distance_squared <= 1)  // 若在單位圓內
            number_in_circle++;
    }

    if (world_rank == 0)
    {
        // ============================================
        // Root Process (rank 0) - 資料收集者
        // ============================================
        
        // 分配用於存放所有 processes 結果的陣列
        long long int *all_counts;
        
        // 使用 MPI_Alloc_mem 而非 malloc：為 RMA 優化的記憶體分配
        // 某些 MPI 實作可能對這種記憶體有特殊優化（如 DMA、RDMA）
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &all_counts);
        
        // 檢查記憶體分配是否成功
        if (all_counts == NULL)
        {
            fprintf(stderr, "rank 0: MPI_Alloc_mem failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // 初始化陣列為 0（雖然後續會被覆蓋，但這是好習慣）
        for (int i = 0; i < world_size; i++)
        {
            all_counts[i] = 0;
        }
        
        // 建立 RMA 窗口：將 all_counts 暴露給其他 processes
        // 參數說明：
        //   - all_counts: 要暴露的記憶體區域
        //   - world_size * sizeof(long long int): 記憶體區域大小（位元組）
        //   - sizeof(long long int): displacement unit（偏移單位）
        //     當 workers 指定 target_disp=rank 時，實際偏移 = rank * sizeof(long long int)
        //   - MPI_INFO_NULL: 不使用額外的提示資訊
        //   - MPI_COMM_WORLD: 在這個 communicator 上建立窗口
        //   - &win: 輸出的窗口物件
        MPI_Win_create(all_counts, world_size * sizeof(long long int), sizeof(long long int),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        
        // Root 自己的結果直接寫入（不需要透過 RMA）
        all_counts[0] = number_in_circle;
        
        // ============================================
        // RMA 同步：兩次 fence 的作用
        // ============================================
        
        // 第一次 fence：開啟 RMA epoch
        // 作用：
        //   1. 等待所有 processes 準備好（集體同步）
        //   2. 開始一個新的 RMA epoch，允許 workers 進行 MPI_Accumulate
        MPI_Win_fence(0, win);
        
        // 在這兩次 fence 之間，workers 會執行 MPI_Accumulate
        // 將它們的 number_in_circle 寫入 all_counts[rank]
        
        // 第二次 fence：結束 RMA epoch
        // 作用：
        //   1. 等待所有 workers 的 MPI_Accumulate 操作完成
        //   2. 確保所有寫入的資料對 root 可見（記憶體一致性）
        //   3. 與所有 processes 同步
        MPI_Win_fence(0, win);
        
        // 現在可以安全地讀取 all_counts，因為所有 RMA 操作都已完成
        
        // ============================================
        // 計算 π 的估計值
        // ============================================
        long long int total_in_circle = 0;
        
        // 累加所有 processes 的結果
        for (int i = 0; i < world_size; i++)
        {
            total_in_circle += all_counts[i];
        }
        
        // π = 4 * (圓內點數 / 總點數)
        // 原理：圓面積 / 正方形面積 = π/4
        pi_result = 4.0 * total_in_circle / tosses;
        
        // 釋放 RMA 記憶體（使用 MPI_Free_mem 而非 free）
        MPI_Free_mem(all_counts);
    }
    else
    {
        // ============================================
        // Worker Processes (rank > 0) - 資料傳送者
        // ============================================
        
        // Workers 不需要提供記憶體，因為它們只是寫入 root 的記憶體
        // 但仍需建立窗口以參與 RMA 通訊（集體操作）
        // 參數說明：
        //   - NULL: 不暴露任何本地記憶體
        //   - 0: 記憶體大小為 0
        //   - 1: displacement unit（此處無意義，因為沒有記憶體）
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        
        // ============================================
        // RMA 同步與資料傳送
        // ============================================
        
        // 第一次 fence：開啟 RMA epoch
        // 作用：與 root 及其他 workers 同步，準備進行 RMA 操作
        MPI_Win_fence(0, win);
        
        // 使用 MPI_Accumulate 將本 process 的結果寫入 root 的 all_counts[world_rank]
        // 參數說明：
        //   - &number_in_circle: 要傳送的資料（origin buffer）
        //   - 1: 傳送 1 個元素
        //   - MPI_LONG_LONG: 資料型別
        //   - 0: 目標 process 的 rank（root）
        //   - world_rank: 目標記憶體的位移（displacement）
        //     實際位置 = world_rank * sizeof(long long int)
        //     因此寫入 all_counts[world_rank]
        //   - 1: 目標端的元素數量
        //   - MPI_LONG_LONG: 目標端的資料型別
        //   - MPI_REPLACE: 操作模式（直接覆蓋，不做加總等運算）
        //   - win: 使用的窗口物件
        MPI_Accumulate(&number_in_circle, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG,
                       MPI_REPLACE, win);
        
        // 第二次 fence：結束 RMA epoch
        // 作用：
        //   1. 確保本 process 的 MPI_Accumulate 操作完成
        //   2. 與 root 及其他 workers 同步
        //   3. 等待所有 RMA 操作結束
        MPI_Win_fence(0, win);
    }

    // 釋放窗口物件（所有 processes 都必須呼叫）
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        // PI result is already calculated above

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

