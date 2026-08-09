#pragma once

#include <tuple>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/equal.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include "torch_thrust_execution.h"

#ifdef __CUDACC__
#define MST_HOST_DEVICE __host__ __device__
#else
#define MST_HOST_DEVICE
#endif

namespace MST_boruvka {

namespace detail {

// Iterative pointer-doubling path compression.
// After completion, succ[i] == root of component containing i.
template<template<typename> class VectorType>
void path_compress(VectorType<int>& succ, const int n)
{
    VectorType<int> succ_temp(n);
    for (int iter = 0; iter < 32; ++iter) {
        // succ_temp[i] = succ[succ[i]]
        thrust::gather(RAMA_THRUST_EXEC succ.begin(), succ.begin() + n, succ.begin(), succ_temp.begin());
        if (thrust::equal(RAMA_THRUST_EXEC succ.begin(), succ.begin() + n, succ_temp.begin()))
            break;
        succ.swap(succ_temp);
    }
}

} // namespace detail

// Boruvka's algorithm for maximum spanning tree.
// Input: symmetric edges (both directions: i->j and j->i with same cost).
// Output: single-direction MST edges (one per undirected MST edge).
template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>, VectorType<float>>
maximum_spanning_tree(
    const VectorType<int>& tails,
    const VectorType<int>& heads,
    const VectorType<float>& costs)
{
    const int n_directed = tails.size();
    assert(n_directed == (int)heads.size());
    assert(n_directed == (int)costs.size());

    if (n_directed == 0)
        return {VectorType<int>(), VectorType<int>(), VectorType<float>()};

    // Input is already symmetric (both directions present).
    // Assign edge IDs: for each directed edge, the ID is its index in
    // the canonical direction (tail < head). Both directions of the same
    // undirected edge get the same ID so dedup works at the end.
    VectorType<int> u(tails.begin(), tails.end());
    VectorType<int> v(heads.begin(), heads.end());
    VectorType<float> w(n_directed);
    VectorType<int> id(n_directed);

    // Negate costs for minimum spanning tree on negated weights = maximum spanning tree.
    thrust::transform(RAMA_THRUST_EXEC costs.begin(), costs.end(), w.begin(), thrust::negate<float>());

    // Assign canonical edge IDs: for edge (a,b), ID = index of the direction where a < b.
    // We build a lookup: sort edges by (min,max) to pair directions, then assign.
    // Simple approach: ID = own index. Dedup at end handles both directions.
    thrust::sequence(RAMA_THRUST_EXEC id.begin(), id.end());

    const int n_vertices_init = thrust::max(*thrust::max_element(RAMA_THRUST_EXEC u.begin(), u.end()) + 1, *thrust::max_element(RAMA_THRUST_EXEC v.begin(), v.end()) + 1);
    int n_vertices = n_vertices_init;
    int n_edges = n_directed;

    // Temp buffers
    VectorType<int> u_tmp(n_edges), v_tmp(n_edges), id_tmp(n_edges);
    VectorType<float> w_tmp(n_edges);

    VectorType<int> succ(n_vertices);
    VectorType<int> succ_id(n_vertices);
    VectorType<int> succ_input(n_vertices);  // input buffer for remove_circles
    VectorType<int> succ_temp(n_vertices);
    VectorType<int> indices(n_edges);
    VectorType<int> flags(n_edges);

    // Collected MST edge ids (at most n_vertices - 1 undirected MST edges,
    // but may collect both directions before dedup)
    VectorType<int> mst_edge_ids(n_directed);
    int n_mst = 0;

    // Reduce-by-key output buffers
    VectorType<int> rbk_keys(n_vertices);
    VectorType<float> rbk_w(n_vertices);
    VectorType<int> rbk_v(n_vertices);
    VectorType<int> rbk_id(n_vertices);

    while (true) {
        if (n_edges == 0)
            break;

        if (n_edges == 1) {
            // Single remaining edge is always part of MST
            VectorType<int> single_id(1);
            thrust::copy(RAMA_THRUST_EXEC id.begin(), id.begin() + 1, single_id.begin());
            int eid = single_id[0];
            mst_edge_ids[n_mst++] = eid;
            break;
        }

        // Steps 1-2: Sort by (source, weight, destination), then copy the first
        // edge for each source. This avoids tuple reduce_by_key, which reads
        // invalid tuple storage with CUDA 13 on sm_120.
        thrust::sort_by_key(RAMA_THRUST_EXEC
            thrust::make_zip_iterator(thrust::make_tuple(u.begin(), w.begin(), v.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin() + n_edges, w.begin() + n_edges, v.begin() + n_edges)),
            id.begin());
        {
            const int* u_ptr = thrust::raw_pointer_cast(u.data());
            int* flags_ptr = thrust::raw_pointer_cast(flags.data());
            thrust::for_each(RAMA_THRUST_EXEC
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(n_edges),
                [u_ptr, flags_ptr] MST_HOST_DEVICE (int pos) {
                    flags_ptr[pos] = (pos == 0 || u_ptr[pos] != u_ptr[pos - 1]) ? 1 : 0;
                });
        }
        thrust::exclusive_scan(RAMA_THRUST_EXEC
            flags.begin(), flags.begin() + n_edges, indices.begin());
        VectorType<int> last_scan(1), last_flag(1);
        thrust::copy(RAMA_THRUST_EXEC
            indices.begin() + n_edges - 1, indices.begin() + n_edges, last_scan.begin());
        thrust::copy(RAMA_THRUST_EXEC
            flags.begin() + n_edges - 1, flags.begin() + n_edges, last_flag.begin());
        const int n_min_edges = last_scan[0] + last_flag[0];
        thrust::scatter_if(RAMA_THRUST_EXEC
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin(), w.begin(), v.begin(), id.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin() + n_edges, w.begin() + n_edges, v.begin() + n_edges, id.begin() + n_edges)),
            indices.begin(), flags.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(
                rbk_keys.begin(), rbk_w.begin(), rbk_v.begin(), rbk_id.begin())));

        // Step 3: Build successor pointers
        // succ_input[vertex] = destination of min-weight edge from vertex
        // succ_id[vertex] = original edge id of that edge
        thrust::sequence(RAMA_THRUST_EXEC succ_input.begin(), succ_input.begin() + n_vertices);
        // Initialize succ_id to -1
        thrust::fill(RAMA_THRUST_EXEC succ_id.begin(), succ_id.begin() + n_vertices, -1);

        thrust::scatter(RAMA_THRUST_EXEC 
            thrust::make_zip_iterator(thrust::make_tuple(
                rbk_v.begin(), rbk_id.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                rbk_v.begin() + n_min_edges, rbk_id.begin() + n_min_edges)),
            rbk_keys.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(
                succ_input.begin(), succ_id.begin())));

        // Step 4: Break 2-cycles
        // If vertex i points to j and j points to i, break by keeping only
        // the edge from the smaller vertex.
        {
            const int n = n_vertices;
            const int* si_ptr = thrust::raw_pointer_cast(succ_input.data());
            int* succ_ptr = thrust::raw_pointer_cast(succ.data());
            int* aux_ptr = thrust::raw_pointer_cast(succ_temp.data());

            thrust::for_each(RAMA_THRUST_EXEC 
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(n),
                [si_ptr, succ_ptr, aux_ptr] MST_HOST_DEVICE (int pos) {
                    int successor = si_ptr[pos];
                    int s_successor = si_ptr[successor];
                    successor = ((successor > pos) && (s_successor == pos)) ? pos : successor;
                    aux_ptr[pos] = (successor != pos) ? 1 : 0;
                    succ_ptr[pos] = successor;
                });
        }

        // Step 5: Collect MST edges
        // aux[i] == 1 means vertex i has a new MST edge
        thrust::exclusive_scan(RAMA_THRUST_EXEC succ_temp.begin(), succ_temp.begin() + n_vertices,
            succ_input.begin()); // reuse succ_input as scan output

        thrust::scatter_if(RAMA_THRUST_EXEC 
            succ_id.begin(), succ_id.begin() + n_vertices,
            succ_input.begin(),
            succ_temp.begin(),
            mst_edge_ids.begin() + n_mst);

        {
            VectorType<int> last_scan(1), last_flag(1);
            thrust::copy(RAMA_THRUST_EXEC succ_input.begin() + n_vertices - 1, succ_input.begin() + n_vertices, last_scan.begin());
            thrust::copy(RAMA_THRUST_EXEC succ_temp.begin() + n_vertices - 1, succ_temp.begin() + n_vertices, last_flag.begin());
            n_mst += last_scan[0] + last_flag[0];
        }

        // Step 6: Path compression to find connected components
        // succ[i] is the representative of i's component
        thrust::sequence(RAMA_THRUST_EXEC succ_input.begin(), succ_input.begin() + n_vertices);
        detail::path_compress<VectorType>(succ, n_vertices);

        // Step 7: Assign new contiguous component IDs
        // Sort vertices by component representative, then assign dense IDs
        thrust::sequence(RAMA_THRUST_EXEC succ_input.begin(), succ_input.begin() + n_vertices);
        thrust::sort_by_key(RAMA_THRUST_EXEC succ.begin(), succ.begin() + n_vertices, succ_input.begin());

        // Mark segment boundaries
        {
            const int n = n_vertices;
            const int* sorted_ptr = thrust::raw_pointer_cast(succ.data());
            int* marks_ptr = thrust::raw_pointer_cast(succ_temp.data());

            thrust::for_each(RAMA_THRUST_EXEC 
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(n),
                [sorted_ptr, marks_ptr, n] MST_HOST_DEVICE (int pos) {
                    marks_ptr[pos] = ((pos == n - 1) || (sorted_ptr[pos] != sorted_ptr[pos + 1])) ? 1 : 0;
                });
        }

        // new_vertices maps old vertex -> new component id
        VectorType<int>& new_vertices = succ; // reuse succ buffer
        thrust::exclusive_scan(RAMA_THRUST_EXEC succ_temp.begin(), succ_temp.begin() + n_vertices,
            succ_id.begin()); // reuse succ_id as scan output
        thrust::scatter(RAMA_THRUST_EXEC succ_id.begin(), succ_id.begin() + n_vertices,
            succ_input.begin(), new_vertices.begin());

        int new_n_vertices;
        {
            VectorType<int> last_scan(1), last_flag(1);
            thrust::copy(RAMA_THRUST_EXEC succ_id.begin() + n_vertices - 1, succ_id.begin() + n_vertices, last_scan.begin());
            thrust::copy(RAMA_THRUST_EXEC succ_temp.begin() + n_vertices - 1, succ_temp.begin() + n_vertices, last_flag.begin());
            new_n_vertices = last_scan[0] + last_flag[0];
        }

        // Step 8: Filter inter-component edges
        {
            const int* nv_ptr = thrust::raw_pointer_cast(new_vertices.data());
            const int* u_ptr = thrust::raw_pointer_cast(u.data());
            const int* v_ptr_data = thrust::raw_pointer_cast(v.data());
            int* flags_ptr = thrust::raw_pointer_cast(flags.data());
            const int ne = n_edges;

            thrust::for_each(RAMA_THRUST_EXEC 
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(ne),
                [nv_ptr, u_ptr, v_ptr_data, flags_ptr] MST_HOST_DEVICE (int pos) {
                    flags_ptr[pos] = (nv_ptr[u_ptr[pos]] != nv_ptr[v_ptr_data[pos]]) ? 1 : 0;
                });
        }

        thrust::exclusive_scan(RAMA_THRUST_EXEC flags.begin(), flags.begin() + n_edges, indices.begin());

        int new_n_edges;
        {
            VectorType<int> last_scan(1), last_flag(1);
            thrust::copy(RAMA_THRUST_EXEC indices.begin() + n_edges - 1, indices.begin() + n_edges, last_scan.begin());
            thrust::copy(RAMA_THRUST_EXEC flags.begin() + n_edges - 1, flags.begin() + n_edges, last_flag.begin());
            new_n_edges = last_scan[0] + last_flag[0];
        }

        if (new_n_edges == 0)
            break;

        // Compact edges
        thrust::scatter_if(RAMA_THRUST_EXEC 
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin(), v.begin(), w.begin(), id.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                u.begin() + n_edges, v.begin() + n_edges, w.begin() + n_edges, id.begin() + n_edges)),
            indices.begin(), flags.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(
                u_tmp.begin(), v_tmp.begin(), w_tmp.begin(), id_tmp.begin())));

        // Step 9: Relabel endpoints with new component IDs. Gather cannot write
        // in-place because earlier outputs may overwrite later input indices.
        thrust::gather(RAMA_THRUST_EXEC u_tmp.begin(), u_tmp.begin() + new_n_edges, new_vertices.begin(), u.begin());
        thrust::gather(RAMA_THRUST_EXEC v_tmp.begin(), v_tmp.begin() + new_n_edges, new_vertices.begin(), v.begin());

        // Step 10: Swap buffers and repeat
        w.swap(w_tmp);
        id.swap(id_tmp);

        n_vertices = new_n_vertices;
        n_edges = new_n_edges;

        // Resize temp buffers for potentially smaller problem
        if ((int)succ.size() < n_vertices) {
            succ.resize(n_vertices);
            succ_id.resize(n_vertices);
            succ_input.resize(n_vertices);
            succ_temp.resize(n_vertices);
            rbk_keys.resize(n_vertices);
            rbk_w.resize(n_vertices);
            rbk_v.resize(n_vertices);
            rbk_id.resize(n_vertices);
        }
    }

    // Gather MST edges from original input
    VectorType<int> mst_tails(n_mst), mst_heads(n_mst);
    VectorType<float> mst_costs(n_mst);
    thrust::gather(RAMA_THRUST_EXEC mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst,
        tails.begin(), mst_tails.begin());
    thrust::gather(RAMA_THRUST_EXEC mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst,
        heads.begin(), mst_heads.begin());
    thrust::gather(RAMA_THRUST_EXEC mst_edge_ids.begin(), mst_edge_ids.begin() + n_mst,
        costs.begin(), mst_costs.begin());

    // Normalize to canonical direction (tail < head) for dedup
    {
        const int nm = n_mst;
        int* t_ptr = thrust::raw_pointer_cast(mst_tails.data());
        int* h_ptr = thrust::raw_pointer_cast(mst_heads.data());

        thrust::for_each(RAMA_THRUST_EXEC 
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(nm),
            [t_ptr, h_ptr] MST_HOST_DEVICE (int i) {
                if (t_ptr[i] > h_ptr[i]) {
                    int tmp = t_ptr[i];
                    t_ptr[i] = h_ptr[i];
                    h_ptr[i] = tmp;
                }
            });
    }

    // Sort by (tail, head) and deduplicate
    VectorType<int> dedup_indices(n_mst);
    thrust::sequence(RAMA_THRUST_EXEC dedup_indices.begin(), dedup_indices.end());
    thrust::sort_by_key(RAMA_THRUST_EXEC 
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin(), mst_heads.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin() + n_mst, mst_heads.begin() + n_mst)),
        thrust::make_zip_iterator(thrust::make_tuple(mst_costs.begin(), dedup_indices.begin())));

    auto new_end = thrust::unique_by_key(RAMA_THRUST_EXEC 
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin(), mst_heads.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin() + n_mst, mst_heads.begin() + n_mst)),
        thrust::make_zip_iterator(thrust::make_tuple(mst_costs.begin(), dedup_indices.begin())));

    int n_unique = new_end.first -
        thrust::make_zip_iterator(thrust::make_tuple(mst_tails.begin(), mst_heads.begin()));

    mst_tails.resize(n_unique);
    mst_heads.resize(n_unique);
    mst_costs.resize(n_unique);

    return {std::move(mst_tails), std::move(mst_heads), std::move(mst_costs)};
}

} // namespace MST_boruvka
