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

    // TODO: MPI init
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

    if (world_rank > 0)
    {
        // TODO: MPI workers
        // Use blocking send 
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

        // MPI_Request requests[];
        long long int total_in_circle = number_in_circle;
        // MPI _Irecv requires separate buffers for each receive
        long long int *received_counts = (long long int *)malloc((world_size - 1) * sizeof(long long int));
        MPI_Request *requests = (MPI_Request *)malloc((world_size - 1) * sizeof(MPI_Request));
        
        // Issue all non-blocking receives
        for (int i = 1; i < world_size; i++)
        {
            MPI_Irecv(&received_counts[i - 1], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        // Wait for each receive individually and accumulate immediately
        // This allows overlap of communication and computation
        for (int i = 0; i < world_size - 1; i++)
        {
            MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
            total_in_circle += received_counts[i];
        }
        
        pi_result = 4.0 * total_in_circle / tosses;
        
        free(received_counts);
        free(requests);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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

