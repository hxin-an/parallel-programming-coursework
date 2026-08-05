#define _GNU_SOURCE // Needed for pthread_setaffinity_np
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>    // For memcpy
#include <pthread.h>
#include <time.h>
#include <immintrin.h> // For AVX, AVX2, FMA
#include <sched.h>     // For sched_setaffinity
#include <unistd.h>

#if defined(__GNUC__) || defined(__clang__)
#define ALIGNED(x) __attribute__((aligned(x)))
#else
#include <intrin.h>
#define ALIGNED(x) __declspec(align(x))
#endif

#define VEC_SIZE 8

// UNROLL_FACTOR: 4, 8, 16 are worth testing on the target machine.
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 16
#endif

#define TOSSES_PER_ITER (VEC_SIZE * UNROLL_FACTOR)

// Aligned struct for thread arguments to prevent false sharing.
typedef struct ALIGNED(64) {
    long long int tosses_per_thread;
    __m256i prng_state_x;
    __m256i prng_state_y;
    long long int hits;
    int thread_id;
    char padding[44]; // Pad to 128 bytes to guarantee separation of cache lines
} thread_arg_t;

// Vectorized xorshift32 PRNG.
static inline __m256i xorshift32_avx2(__m256i* state) {
    __m256i x = *state;
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
    *state = x;
    return x;
}

// Generates a vector of 8 random floats in [0, 1) using the mantissa method
static inline __m256 sim_rand_ps(__m256i* state) {
    __m256i random_u32 = xorshift32_avx2(state);
    const __m256i one_int_bits = _mm256_set1_epi32(0x3F800000);
    const __m256 one_float_vec = _mm256_set1_ps(1.0f);
    __m256i random_23_bits = _mm256_srli_epi32(random_u32, 9);
    __m256 random_one_two = _mm256_castsi256_ps(_mm256_or_si256(random_23_bits, one_int_bits));
    return _mm256_sub_ps(random_one_two, one_float_vec);
}

#ifndef USE_POPCNT_PATH
// Horizontal sum for a __m256i vector (for vector-add path)
static inline int horizontal_add_epi32(__m256i vec) {
    __m128i lo = _mm256_castsi256_si128(vec);
    __m128i hi = _mm256_extracti128_si256(vec, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}
#endif

void* worker(void* arg) {
    // Set FTZ and DAZ flags to avoid denormal penalty
    _mm_setcsr(_mm_getcsr() | 0x8040);

    thread_arg_t* thread_arg = (thread_arg_t*)arg;
    
    // Localize PRNG state to prevent false sharing by ensuring updates happen
    // in registers instead of memory shared across threads.
    __m256i local_prng_state_x = thread_arg->prng_state_x;
    __m256i local_prng_state_y = thread_arg->prng_state_y;

    const __m256 one_vec = _mm256_set1_ps(1.0f);
    long long int tosses = thread_arg->tosses_per_thread;
    long long int vec_tosses = tosses - (tosses % TOSSES_PER_ITER);
    long long int total_hits = 0;

#ifdef USE_POPCNT_PATH
    // POPCNT path: sum hits scalarilly per unrolled iteration
    for (long long int i = 0; i < vec_tosses; i += TOSSES_PER_ITER) {
        #define ITERATION(j) \
            __m256 x##j = sim_rand_ps(&local_prng_state_x); \
            __m256 y##j = sim_rand_ps(&local_prng_state_y); \
            __m256 dist_sq##j = _mm256_fmadd_ps(y##j, y##j, _mm256_mul_ps(x##j, x##j)); \
            __m256 cmp##j = _mm256_cmp_ps(dist_sq##j, one_vec, _CMP_LE_OQ); \
            total_hits += _mm_popcnt_u32(_mm256_movemask_ps(cmp##j));

        ITERATION(0); ITERATION(1); ITERATION(2); ITERATION(3);
        ITERATION(4); ITERATION(5); ITERATION(6); ITERATION(7);
        ITERATION(8); ITERATION(9); ITERATION(10); ITERATION(11);
        ITERATION(12); ITERATION(13); ITERATION(14); ITERATION(15);
        #undef ITERATION
    }
#else
    // Default path: vector accumulator
    __m256i hits_vec = _mm256_setzero_si256();
    const __m256i one_epi32 = _mm256_set1_epi32(1);
    for (long long int i = 0; i < vec_tosses; i += TOSSES_PER_ITER) {
        #define ITERATION(j) \
            __m256 x##j = sim_rand_ps(&local_prng_state_x); \
            __m256 y##j = sim_rand_ps(&local_prng_state_y); \
            __m256 dist_sq##j = _mm256_fmadd_ps(y##j, y##j, _mm256_mul_ps(x##j, x##j)); \
            __m256 cmp##j = _mm256_cmp_ps(dist_sq##j, one_vec, _CMP_LE_OQ); \
            __m256i hits_mask##j = _mm256_and_si256(_mm256_castps_si256(cmp##j), one_epi32); \
            hits_vec = _mm256_add_epi32(hits_vec, hits_mask##j);

        ITERATION(0); ITERATION(1); ITERATION(2); ITERATION(3);
        ITERATION(4); ITERATION(5); ITERATION(6); ITERATION(7);
        ITERATION(8); ITERATION(9); ITERATION(10); ITERATION(11);
        ITERATION(12); ITERATION(13); ITERATION(14); ITERATION(15);
        #undef ITERATION
    }
    total_hits = horizontal_add_epi32(hits_vec);
#endif

    // Handle remaining tosses with a scalar loop
    long long int remaining_tosses = tosses % TOSSES_PER_ITER;
    if (remaining_tosses > 0) {
        uint32_t rand_state_x, rand_state_y;
        // Copy the latest PRNG state from local vars for the scalar loop
        memcpy(&rand_state_x, &local_prng_state_x, sizeof(uint32_t));
        memcpy(&rand_state_y, &local_prng_state_y, sizeof(uint32_t));

        for (long long int i = 0; i < remaining_tosses; ++i) {
            rand_state_x ^= rand_state_x << 13; rand_state_x ^= rand_state_x >> 17; rand_state_x ^= rand_state_x << 5;
            rand_state_y ^= rand_state_y << 13; rand_state_y ^= rand_state_y >> 17; rand_state_y ^= rand_state_y << 5;
            
            uint32_t temp_x = (rand_state_x >> 9) | 0x3F800000;
            uint32_t temp_y = (rand_state_y >> 9) | 0x3F800000;
            float x, y;
            memcpy(&x, &temp_x, sizeof(float));
            memcpy(&y, &temp_y, sizeof(float));
            x -= 1.0f;
            y -= 1.0f;

            if (x * x + y * y <= 1.0f) {
                total_hits++;
            }
        }
    }

    thread_arg->hits = total_hits;
    pthread_exit(NULL);
}

// Simple hash for seeding
uint32_t hash_seed(uint32_t seed, uint32_t tid) {
    seed ^= tid * 0x9e3779b9; // Gold number
    seed ^= seed >> 16;
    seed ^= seed * 0x85ebca6b;
    seed ^= seed >> 13;
    seed ^= seed * 0xc2b2ae35;
    seed ^= seed >> 16;
    return seed;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads> <num_tosses>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long long int total_tosses = atoll(argv[2]);

    if (num_threads <= 0 || total_tosses <= 0) {
        fprintf(stderr, "Number of threads and tosses must be positive.\n");
        return 1;
    }

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_arg_t* args = _mm_malloc(num_threads * sizeof(thread_arg_t), 64);
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    long long int tosses_per_thread_base = (total_tosses / num_threads) / TOSSES_PER_ITER * TOSSES_PER_ITER;
    long long int leftover_tosses = total_tosses - (tosses_per_thread_base * num_threads);
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    (void)num_cpus; // Suppress unused variable warning if PIN_THREADS is not defined

    uint32_t time_seed = 20241015;

    for (int i = 0; i < num_threads; i++) {
        args[i].tosses_per_thread = tosses_per_thread_base;
        if (i == num_threads - 1) {
            args[i].tosses_per_thread += leftover_tosses;
        }
        args[i].hits = 0;
        args[i].thread_id = i;

        uint32_t thread_seed = hash_seed(time_seed, i);
        
        ALIGNED(32) uint32_t seed_data_x[VEC_SIZE];
        ALIGNED(32) uint32_t seed_data_y[VEC_SIZE];
        for (int k = 0; k < VEC_SIZE; ++k) {
            seed_data_x[k] = hash_seed(thread_seed, k);
            seed_data_y[k] = hash_seed(thread_seed, k + VEC_SIZE);
        }
        args[i].prng_state_x = _mm256_load_si256((__m256i*)seed_data_x);
        args[i].prng_state_y = _mm256_load_si256((__m256i*)seed_data_y);

        // Set CPU affinity in thread attributes
#ifdef PIN_THREADS
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i % num_cpus, &cpuset);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
#endif

        pthread_create(&threads[i], &attr, worker, &args[i]);
    }

    long long int total_hits = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_hits += args[i].hits;
    }

    pthread_attr_destroy(&attr);
    double pi_estimate = 4.0 * (double)total_hits / (double)total_tosses;
    printf("%1f\n", pi_estimate);

    free(threads);
    _mm_free(args);

    return 0;
}
