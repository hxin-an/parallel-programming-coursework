#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

// Clear the vertex set
void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

// Initialize the vertex set with a given maximum size
void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}


// Free the memory allocated for the vertex set
void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Perform one step of the top-down BFS
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances, int current_distance)
{
#pragma omp parallel
    {
        std::vector<int> local_new_frontier;

#pragma omp for schedule(dynamic, 1024)
        // Explore each node in the current frontier
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER)
                {
                    if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, current_distance + 1))
                    {
                        local_new_frontier.push_back(outgoing);
                    }
                }
            }
        }

        if (!local_new_frontier.empty())
        {
            int start_index = __sync_fetch_and_add(&new_frontier->count, local_new_frontier.size());
            // Update the new frontier

//#pragma omp parallel for
            for (size_t i = 0; i < local_new_frontier.size(); ++i)
            {
                new_frontier->vertices[start_index + i] = local_new_frontier[i];
            }
        }
    }
}

void bfs_top_down(Graph graph, solution *sol)
{
    // Initialize the two vertex sets
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    // Pointers to the current and new frontier
    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

//#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Initialize the BFS

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int current_distance = 0;

    while (frontier->count != 0)
    {
        // Perform the BFS step
#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);
        top_down_step(graph, frontier, new_frontier, sol->distances, current_distance);
//      printf("New frontier size: %d\n", new_frontier->count);
#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
    // Swap frontiers
        current_distance++;
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // Initialize the two vertex sets
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    bool *frontier_bitmap = new bool[graph->num_nodes];
    // Initialize the bitmap
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    // Set the root in the bitmap
    int current_distance = 0;
    // BFS loop
    while (frontier->count != 0)
    {
        // Build the frontier bitmap
#pragma omp parallel for
        for (int i = 0; i < graph->num_nodes; i++)
            frontier_bitmap[i] = false;
#pragma omp parallel for
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            frontier_bitmap[node] = true;
        }

        vertex_set_clear(new_frontier);
        // Bottom-up step

#pragma omp parallel
        {
            std::vector<int> local_new_frontier;
#pragma omp for schedule(dynamic, 1024)
            for (int i = 0; i < graph->num_nodes; i++)
            {
                if (sol->distances[i] == NOT_VISITED_MARKER)
                {
                    const Vertex *start = incoming_begin(graph, i);
                    const Vertex *end = incoming_end(graph, i);
                    for (const Vertex *v = start; v != end; ++v)
                    {
                        int neighbor = *v;
                        if (frontier_bitmap[neighbor])
                        {
                            sol->distances[i] = current_distance + 1;
                            local_new_frontier.push_back(i);
                            break;
                        }
                    }
                }
            }
            // Update the new frontier
            if (!local_new_frontier.empty())
            {
                int start_index = __sync_fetch_and_add(&new_frontier->count, local_new_frontier.size());
//#pragma omp parallel for
                for (size_t i = 0; i < local_new_frontier.size(); ++i)
                {
                    new_frontier->vertices[start_index + i] = local_new_frontier[i];
                }
            }
        }
        // Swap frontiers
        current_distance++;
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
    // Cleanup
    delete[] frontier_bitmap;
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_hybrid(Graph graph, solution *sol)
{
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    bool *frontier_bitmap = new bool[graph->num_nodes];
    // Initialize the bitmap
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    // Initialize the BFS
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    // Set the root in the bitmap
    int current_distance = 0;
    bool is_bottom_up = false;
    const int top_down_threshold = graph->num_nodes / 24;
    const int bottom_up_threshold = graph->num_nodes / 12;

    while (frontier->count != 0)
    {
        if (!is_bottom_up && frontier->count > bottom_up_threshold)
        {
            is_bottom_up = true;
        }
        else if (is_bottom_up && frontier->count < top_down_threshold)
        {
            is_bottom_up = false;
        }

        vertex_set_clear(new_frontier);

        if (is_bottom_up)
        {
#pragma omp parallel for
            for (int i = 0; i < graph->num_nodes; i++)
                frontier_bitmap[i] = false;
            // Build the frontier bitmap
#pragma omp parallel for
            for (int i = 0; i < frontier->count; i++)
            {
                int node = frontier->vertices[i];
                frontier_bitmap[node] = true;
            }
// Bottom-up step
#pragma omp parallel
            {
                std::vector<int> local_new_frontier;
#pragma omp for schedule(dynamic, 1024)
                // Explore each node in the current frontier
                for (int i = 0; i < graph->num_nodes; i++)
                {
                    if (sol->distances[i] == NOT_VISITED_MARKER)
                    {
                        const Vertex *start = incoming_begin(graph, i);
                        const Vertex *end = incoming_end(graph, i);
                        // Check incoming edges
                        for (const Vertex *v = start; v != end; ++v)
                        {
                            int neighbor = *v;
                            if (frontier_bitmap[neighbor])
                            {
                                sol->distances[i] = current_distance + 1;
                                local_new_frontier.push_back(i);
                                break;
                            }
                        }
                    }
                }
                // Update the new frontier
                if (!local_new_frontier.empty())
                {
                    int start_index = __sync_fetch_and_add(&new_frontier->count, local_new_frontier.size());
                    for (size_t i = 0; i < local_new_frontier.size(); ++i)
                    {
                        new_frontier->vertices[start_index + i] = local_new_frontier[i];
                    }
                }
            }
        }
        else
        {
            top_down_step(graph, frontier, new_frontier, sol->distances, current_distance);
        }

        current_distance++;
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    delete[] frontier_bitmap;
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}


