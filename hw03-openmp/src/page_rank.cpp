#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

void page_rank(Graph g, double *sol, double damping, double convergence)
{
    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    // adding sink nodes contribution
    // Allocate memory for the new scores
    double *score_new = new double[nnodes];

    // Initialize vertex weights to uniform probability
    #pragma omp parallel for
    for (int i = 0; i < nnodes; ++i)
    {
        sol[i] = equal_prob;
    }

    double diff;
    do
    {
        // Initialize difference to zero
        diff = 0.0;

        // Calculate the contribution of sink nodes (nodes with no outgoing edges)
        double sink_contrib = 0.0;
        #pragma omp parallel for reduction(+:sink_contrib)
        for (int i = 0; i < nnodes; ++i)
        {
            if (outgoing_size(g, i) == 0)
            {
                sink_contrib += sol[i];
            }
        }
        double total_sink_effect = damping * sink_contrib / nnodes;

        // Compute score_new[vi] for all nodes vi
        #pragma omp parallel for
        for (int i = 0; i < nnodes; ++i)
        {
            // Start with zero score
            score_new[i] = 0.0;
            const Vertex *start = incoming_begin(g, i);
            const Vertex *end = incoming_end(g, i);
            for (const Vertex *vj = start; vj != end; ++vj)
            {
                score_new[i] += sol[*vj] / outgoing_size(g, *vj);
            }
            // Apply damping factor and add sink effect
            score_new[i] = (damping * score_new[i]) + (1.0 - damping) / nnodes;
            score_new[i] += total_sink_effect;
        }

        // Compute how much per-node scores have changed and update sol
        // Accumulate the total difference
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < nnodes; ++i)
        {
            diff += std::abs(score_new[i] - sol[i]);
            sol[i] = score_new[i];
        }

    } while (diff >= convergence);



    delete[] score_new;
}

