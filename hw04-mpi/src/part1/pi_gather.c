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

    // Read the MPI world size and current rank.
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    unsigned int seed = (world_rank + 1) * time(NULL);
    long long int local_tosses = tosses / world_size;
    long long int number_in_circle = 0;
    
    // Monte Carlo simulation
    for (long long int toss = 0; toss < local_tosses; toss++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }

    // Gather each process's local hit count at rank zero.
    long long int *all_counts = NULL;
    if (world_rank == 0)
    {
        // Allocate memory to gather results from all processes
        all_counts = (long long int *)malloc(world_size * sizeof(long long int));
    }
    MPI_Gather(&number_in_circle, 1, MPI_LONG_LONG, all_counts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // Aggregate the gathered counts and estimate pi.
        // Aggregate results from all processes
        long long int total_in_circle = 0;
        for (int i = 0; i < world_size; i++)
        {
            total_in_circle += all_counts[i];
        }
        
        pi_result = 4.0 * total_in_circle / tosses;
        
        free(all_counts);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
