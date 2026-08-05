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
    // Initialize MPI communication: get the total number of processes and the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Seed for random number generation
    unsigned int seed = (world_rank + 1) * time(NULL);
    long long int local_tosses = tosses / world_size;

    // Variable to count the number of points that fall inside the unit circle
    long long int number_in_circle = 0;

    // Monte Carlo simulation: generate random points and check if they fall inside the unit circle
    for (long long int toss = 0; toss < local_tosses; toss++)
    {
        // Generate random x and y coordinates in the range [-1, 1]
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;

        // Calculate the squared distance from the origin
        double distance_squared = x * x + y * y;

        // If the point is inside the unit circle, increment the counter
        if (distance_squared <= 1)
            number_in_circle++;
    }

    if (world_rank > 0)
    {
        // Worker processes send their local count of points inside the circle to the root process (rank 0)
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        // printf("Process %d sent %lld points inside circle to root process.\n", world_rank, number_in_circle);
    }
    else if (world_rank == 0)
    {
        // Root process (rank 0) collects results from all worker processes

        // Initialize the total count with the local count of the root process
        long long int total_in_circle = number_in_circle;
        long long int received_count;

        // Receive counts from all other processes and accumulate them
        for (int i = 1; i < world_size; i++)
        {
            MPI_Recv(&received_count, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_in_circle += received_count;
            // printf("Root process received %lld points inside circle from process %d.\n", received_count, i);
        }

        // Calculate the final value of PI using the formula: PI = 4 * (points inside circle) / (total points)
        pi_result = 4.0 * total_in_circle / tosses;
    }

    if (world_rank == 0)
    {
        // --- Don't TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
