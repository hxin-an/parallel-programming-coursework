#include <mpi.h>
#include <cstring>

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    /*
     * construct_matrices:
     * - Distribute rows of A across MPI processes (row-wise block distribution).
     * - Broadcast entire B to all processes (replicate B).
     *
     * Layout choices and rationale (kept simple and cache-friendly):
     * - A is stored in row-major order and each process receives a contiguous block of rows.
     *   This makes accessing A's row elements in the inner loops cheap (good spatial locality).
     * - B is stored in COLUMN-MAJOR order by main.cc: B[k][j] is at b_mat[j * m + k].
     *   This is because main.cc reads B with the pattern: b_mat[(x * m) + y] where x is column index.
     *
     * The function performs the scatter of A manually via point-to-point for clarity.
     */
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Broadcast dimensions to all processes (root provides n,m,l)
    // IMPORTANT: Only rank 0 has valid n,m,l values from main.cc
    int dims[3];
    if (rank == 0) {
        dims[0] = n;
        dims[1] = m;
        dims[2] = l;
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    // All processes update their local copies from broadcasted values
    n = dims[0];
    m = dims[1];
    l = dims[2];
    
    // Row-wise block distribution: each process gets roughly ceil(n/size) rows
    int rows_per_proc = (n + size - 1) / size;
    int start_row = rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    if (end_row > n) end_row = n;
    int local_rows = end_row - start_row;

    // Allocate local block of A (row-major). If local_rows==0, allocation yields size 0 array.
    *a_mat_ptr = new int[local_rows * m];

    // Scatter matrix A from rank 0 to everyone (including rank 0 itself)
    if (rank == 0) {
        // Copy own block (start_row == 0)
        if (local_rows > 0) {
            memcpy(*a_mat_ptr, a_mat, local_rows * m * sizeof(int));
        }

        // Send blocks to other ranks
        for (int r = 1; r < size; r++) {
            int send_start = r * rows_per_proc;
            int send_end = send_start + rows_per_proc;
            if (send_end > n) send_end = n;
            int send_rows = send_end - send_start;

            if (send_rows > 0) {
                MPI_Send(a_mat + send_start * m, send_rows * m, MPI_INT, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Receive local block from root
        if (local_rows > 0) {
            MPI_Recv(*a_mat_ptr, local_rows * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Broadcast entire matrix B (m x l) to all processes. B is replicated on every rank.
    *b_mat_ptr = new int[m * l];
    if (rank == 0) {
        memcpy(*b_mat_ptr, b_mat, m * l * sizeof(int));
    }
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    /*
     * matrix_multiply:
     * - Multiply local block of A (local_rows x m) with B (m x l) and produce local_result
     *   of size local_rows x l.
     * - The implementation uses cache-blocking (tiling) to improve temporal locality and reduce
     *   cache misses. SIMD is disallowed in this environment, so we focus on algorithmic and
     *   cache-friendly optimizations (blocking, loop ordering, pointer hoisting).
     */

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Broadcast dimensions to all processes
    // IMPORTANT: Only rank 0 has valid n,m,l values from main.cc
    int dims[3];
    if (rank == 0) {
        dims[0] = n;
        dims[1] = m;
        dims[2] = l;
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int n_actual = dims[0];
    int m_actual = dims[1];
    int l_actual = dims[2];

    // Compute local row interval (same scheme as construct_matrices)
    int rows_per_proc = (n_actual + size - 1) / size;
    int start_row = rank * rows_per_proc;
    int end_row = start_row + rows_per_proc;
    if (end_row > n_actual) end_row = n_actual;
    int local_rows = end_row - start_row;

    // Allocate local result buffer (initialized to zero)
    int *local_result = new int[local_rows * l_actual];

    // Zero initialize (efficiently)
    memset(local_result, 0, local_rows * l_actual * sizeof(int));

    // Blocked matrix multiplication parameters. BLOCK_SIZE should be tuned to L1/L2 cache
    // characteristics. 32 is a conservative default that often works well on modern CPUs.
    const int BLOCK_SIZE = 32;

    // Since B is stored in column-major format (B[k][j] at b_mat[j*m + k]),
    // we iterate with j in the outer block loop for better cache performance.
    // Loop ordering: jj (column blocks) -> kk (k blocks) -> i (rows) -> j -> k
    // This way, for each column block of B, we access elements more sequentially.

    for (int jj = 0; jj < l_actual; jj += BLOCK_SIZE) {
        int jmax = (jj + BLOCK_SIZE < l_actual) ? jj + BLOCK_SIZE : l_actual;
        
        for (int kk = 0; kk < m_actual; kk += BLOCK_SIZE) {
            int kmax = (kk + BLOCK_SIZE < m_actual) ? kk + BLOCK_SIZE : m_actual;

            // iterate over rows assigned to this process
            for (int i = 0; i < local_rows; i++) {
                const int *a_row = a_mat + i * m_actual;
                int *c_row = local_result + i * l_actual;

                // For each column j in the current block
                for (int j = jj; j < jmax; j++) {
                    int sum = 0;
                    // Accumulate over k in the current k-block
                    // B column j is stored starting at b_mat[j * m_actual]
                    const int *b_col = b_mat + j * m_actual;
                    
                    for (int k = kk; k < kmax; k++) {
                        sum += a_row[k] * b_col[k];
                    }
                    c_row[j] += sum;
                }
            }
        }
    }

    // Gather local results to rank 0 (row-wise)
    if (rank == 0) {
        // copy own result to output
        if (local_rows > 0) {
            memcpy(out_mat, local_result, local_rows * l_actual * sizeof(int));
        }

        // receive blocks from other ranks
        for (int r = 1; r < size; r++) {
            int recv_start = r * rows_per_proc;
            int recv_end = recv_start + rows_per_proc;
            if (recv_end > n_actual) recv_end = n_actual;
            int recv_rows = recv_end - recv_start;

            if (recv_rows > 0) {
                MPI_Recv(out_mat + recv_start * l_actual, recv_rows * l_actual, MPI_INT, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        if (local_rows > 0) {
            MPI_Send(local_result, local_rows * l_actual, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    delete[] local_result;
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    /* Release the input matrices allocated by the harness. */
    delete[] a_mat;
    delete[] b_mat;
}
