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

    // Reduce partial counts through a binary communication tree.
    // Binary tree reduction
    int step = 1;
    while (step < world_size)
    {
        if (world_rank % (2 * step) == 0)
        {
            // Receiver
            if (world_rank + step < world_size)
            {
                // Receive data from the corresponding sender
                long long int received_count;
                MPI_Recv(&received_count, 1, MPI_LONG_LONG, world_rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                number_in_circle += received_count;
                // printf("Process %d received %lld points inside circle from process %d.\n", world_rank, received_count, world_rank + step);
            }       
        }
        else if (world_rank % (2 * step) == step)
        {
            // Sender
            int target = world_rank - step;
            // Send data to the corresponding receiver
            MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, target, 0, MPI_COMM_WORLD);
            break; // This process is done
        }
        step *= 2;
    }

    if (world_rank == 0)
    {
        // Convert the final hit count into an estimate of pi.
        pi_result = 4.0 * number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
