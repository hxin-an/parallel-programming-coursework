#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

struct alignas(64) WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{
    // Cache frequently accessed args members to local variables for minor optimization
    const float x0 = args->x0;
    const float y0 = args->y0;
    const float x1 = args->x1;
    const float y1 = args->y1;
    const unsigned int width = args->width;
    const unsigned int height = args->height;
    const int maxIterations = args->maxIterations;
    int *const output = args->output;
    const int numThreads = args->numThreads;
    const int threadId = args->threadId;

    const int B = 2; // Optimal block size found through tuning

    // The increment in start_row for each thread is constant
    const unsigned int row_increment = numThreads * B;

    for (unsigned int start_row = threadId * B; start_row < height; start_row += row_increment)
    {
        const int num_rows = (start_row + B > height) ? (height - start_row) : B;

        mandelbrot_serial(x0, y0, x1, y1, width, height,
                          start_row, num_rows,
                          maxIterations, output);
    }
}
//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // Copy the shared image parameters into each worker's arguments.
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
}
