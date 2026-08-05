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

    // Sum all local hit counts at rank zero with MPI_Reduce.
    long long int total_in_circle = 0;
    // reduce all local counts into total_in_circle at root process (rank 0)
    MPI_Reduce(&number_in_circle, &total_in_circle, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // Convert the reduced hit count into an estimate of pi.
        pi_result = 4.0 * total_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
