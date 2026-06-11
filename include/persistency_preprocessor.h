#pragma once

#include "connected_components.h"
#include "find_quadrangles.h"
#include "find_triangles.h"
#include "graph.h"
#include "multicut_solver_options.h"
#include "rama_utils.h"
#include "time_measure_util.h"

#include <cassert>
#include <iostream>
#include <tuple>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#ifdef __CUDACC__
#define PREPROCESSOR_HOST_DEVICE __host__ __device__
#else
#define PREPROCESSOR_HOST_DEVICE
#endif

namespace persistency_preprocessor_detail {

constexpr int max_iterations = 100;

struct forward_edge {
    PREPROCESSOR_HOST_DEVICE
    bool operator()(const thrust::tuple<int, int>& edge) const
    {
        return thrust::get<0>(edge) < thrust::get<1>(edge);
    }
};

struct marked_edge {
    PREPROCESSOR_HOST_DEVICE
    bool operator()(const int flag) const
    {
        return flag != 0;
    }
};

PREPROCESSOR_HOST_DEVICE
inline float absolute_value(const float value)
{
    return value < 0.0f ? -value : value;
}

PREPROCESSOR_HOST_DEVICE
inline int find_edge_index(
    const int tail,
    const int head,
    const int* const offsets,
    const int* const heads)
{
    for (int edge = offsets[tail]; edge < offsets[tail + 1]; ++edge)
    {
        if (heads[edge] == head)
            return edge;
    }
    return -1;
}

template<template<typename> class VectorType>
VectorType<float> node_weight_sums(const Graph<VectorType>& graph, const bool positive_only)
{
    const int num_nodes = graph.num_nodes();
    VectorType<float> sums(num_nodes, 0.0f);
    const int num_edges = graph.num_directed_edges();
    if (num_edges == 0)
        return sums;

    VectorType<float> weights(num_edges);
    thrust::transform(
        graph.get_costs().begin(),
        graph.get_costs().end(),
        weights.begin(),
        [positive_only] PREPROCESSOR_HOST_DEVICE (const float cost) {
            if (positive_only)
                return cost > 0.0f ? cost : 0.0f;
            return persistency_preprocessor_detail::absolute_value(cost);
        });

    VectorType<int> unique_tails(num_edges);
    VectorType<float> reduced(num_edges);
    auto end = thrust::reduce_by_key(
        graph.get_tails().begin(),
        graph.get_tails().end(),
        weights.begin(),
        unique_tails.begin(),
        reduced.begin());

    const int num_unique = end.first - unique_tails.begin();
    thrust::scatter(
        reduced.begin(),
        reduced.begin() + num_unique,
        unique_tails.begin(),
        sums.begin());
    return sums;
}

template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>> forward_edges(const Graph<VectorType>& graph)
{
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(graph.get_tails().begin(), graph.get_heads().begin()));
    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(graph.get_tails().end(), graph.get_heads().end()));

    const int count = thrust::count_if(begin, end, forward_edge());
    VectorType<int> tails(count);
    VectorType<int> heads(count);
    auto output = thrust::make_zip_iterator(thrust::make_tuple(tails.begin(), heads.begin()));
    thrust::copy_if(begin, end, output, forward_edge());
    return {std::move(tails), std::move(heads)};
}

template<template<typename> class VectorType>
VectorType<int> edge_criterion_flags(
    const Graph<VectorType>& graph,
    const VectorType<float>& node_costs)
{
    VectorType<int> flags(graph.num_directed_edges(), 0);
    const float* const node_costs_ptr = thrust::raw_pointer_cast(node_costs.data());

    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(graph.get_tails().begin(), graph.get_heads().begin(), graph.get_costs().begin()));
    auto end = thrust::make_zip_iterator(
        thrust::make_tuple(graph.get_tails().end(), graph.get_heads().end(), graph.get_costs().end()));

    thrust::transform(
        begin,
        end,
        flags.begin(),
        [node_costs_ptr] PREPROCESSOR_HOST_DEVICE (const thrust::tuple<int, int, float>& edge) {
            const int tail = thrust::get<0>(edge);
            const int head = thrust::get<1>(edge);
            const float cost = thrust::get<2>(edge);
            return cost > 0.0f
                && ((node_costs_ptr[tail] - 2.0f * cost <= 0.0f)
                    || (node_costs_ptr[head] - 2.0f * cost <= 0.0f));
        });

    return flags;
}

template<template<typename> class VectorType>
VectorType<int> triangle_criterion_edge_indices(
    const Graph<VectorType>& graph,
    const VectorType<float>& node_costs,
    const VectorType<float>& positive_node_costs,
    const bool verbose)
{
    auto [edge_tails, edge_heads] = forward_edges<VectorType>(graph);
    if (edge_tails.empty())
        return VectorType<int>();

    VectorType<int> node_offsets = graph.compute_node_offsets();
    auto [triangle_1, triangle_2, triangle_3] = find_triangles<VectorType>(
        edge_tails,
        edge_heads,
        node_offsets,
        graph.get_heads());
    deduplicate_triangles<VectorType>(triangle_1, triangle_2, triangle_3);

    if (verbose)
        std::cout << "Found: " << triangle_1.size() << " triangles with "
                  << graph.num_edges() << " total edges\n";

    if (triangle_1.empty())
        return VectorType<int>();

    VectorType<int> edge_indices(triangle_1.size() * 6, -1);

    const int* const offsets_ptr = thrust::raw_pointer_cast(node_offsets.data());
    const int* const heads_ptr = graph.get_heads_ptr();
    const float* const costs_ptr = graph.get_costs_ptr();
    const float* const node_costs_ptr = thrust::raw_pointer_cast(node_costs.data());
    const float* const positive_node_costs_ptr = thrust::raw_pointer_cast(positive_node_costs.data());
    const int* const triangle_1_ptr = thrust::raw_pointer_cast(triangle_1.data());
    const int* const triangle_2_ptr = thrust::raw_pointer_cast(triangle_2.data());
    const int* const triangle_3_ptr = thrust::raw_pointer_cast(triangle_3.data());
    int* const edge_indices_ptr = thrust::raw_pointer_cast(edge_indices.data());

    thrust::for_each(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>((int)triangle_1.size()),
        [offsets_ptr, heads_ptr, costs_ptr, node_costs_ptr, positive_node_costs_ptr,
         triangle_1_ptr, triangle_2_ptr, triangle_3_ptr, edge_indices_ptr]
        PREPROCESSOR_HOST_DEVICE (const int triangle) {
            const int u = triangle_1_ptr[triangle];
            const int v = triangle_2_ptr[triangle];
            const int w = triangle_3_ptr[triangle];

            const int uv = persistency_preprocessor_detail::find_edge_index(u, v, offsets_ptr, heads_ptr);
            const int uw = persistency_preprocessor_detail::find_edge_index(u, w, offsets_ptr, heads_ptr);
            const int vw = persistency_preprocessor_detail::find_edge_index(v, w, offsets_ptr, heads_ptr);
            const int vu = persistency_preprocessor_detail::find_edge_index(v, u, offsets_ptr, heads_ptr);
            const int wu = persistency_preprocessor_detail::find_edge_index(w, u, offsets_ptr, heads_ptr);
            const int wv = persistency_preprocessor_detail::find_edge_index(w, v, offsets_ptr, heads_ptr);
            assert(uv >= 0 && uw >= 0 && vw >= 0 && vu >= 0 && wu >= 0 && wv >= 0);

            const float cost_uv = costs_ptr[uv];
            const float cost_uw = costs_ptr[uw];
            const float cost_vw = costs_ptr[vw];
            const float positive_uv = cost_uv > 0.0f ? cost_uv : 0.0f;
            const float positive_uw = cost_uw > 0.0f ? cost_uw : 0.0f;
            const float positive_vw = cost_vw > 0.0f ? cost_vw : 0.0f;
            const float positive_neighborhood =
                (positive_node_costs_ptr[u] + positive_node_costs_ptr[v] + positive_node_costs_ptr[w]) / 2.0f
                - 2.0f * positive_uv - 2.0f * positive_uw - 2.0f * positive_vw;

            if (cost_uv + cost_uw + cost_vw < positive_neighborhood)
                return;

            const float cost_u =
                node_costs_ptr[u]
                - persistency_preprocessor_detail::absolute_value(cost_uv)
                - persistency_preprocessor_detail::absolute_value(cost_uw);
            const float cost_v =
                node_costs_ptr[v]
                - persistency_preprocessor_detail::absolute_value(cost_uv)
                - persistency_preprocessor_detail::absolute_value(cost_vw);
            const float cost_w =
                node_costs_ptr[w]
                - persistency_preprocessor_detail::absolute_value(cost_uw)
                - persistency_preprocessor_detail::absolute_value(cost_vw);

            const bool pred_1 =
                (cost_uv + cost_uw >= cost_u) || (cost_uv + cost_uw >= cost_v + cost_w);
            const bool pred_2 =
                (cost_uw + cost_vw >= cost_w) || (cost_uw + cost_vw >= cost_u + cost_v);
            const bool pred_3 =
                (cost_uv + cost_vw >= cost_v) || (cost_uv + cost_vw >= cost_u + cost_w);

            const int offset = 6 * triangle;
            if (pred_1 || pred_2)
            {
                edge_indices_ptr[offset] = uw;
                edge_indices_ptr[offset + 1] = wu;
            }
            if (pred_1 || pred_3)
            {
                edge_indices_ptr[offset + 2] = uv;
                edge_indices_ptr[offset + 3] = vu;
            }
            if (pred_2 || pred_3)
            {
                edge_indices_ptr[offset + 4] = vw;
                edge_indices_ptr[offset + 5] = wv;
            }
        });

    auto marked_end = thrust::remove(edge_indices.begin(), edge_indices.end(), -1);
    edge_indices.resize(marked_end - edge_indices.begin());
    if (edge_indices.empty())
        return edge_indices;

    thrust::sort(edge_indices.begin(), edge_indices.end());
    auto unique_end = thrust::unique(edge_indices.begin(), edge_indices.end());
    edge_indices.resize(unique_end - edge_indices.begin());
    return edge_indices;
}

template<template<typename> class VectorType>
VectorType<int> contracting_edge_flags(const Graph<VectorType>& graph, const multicut_solver_options& opts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    VectorType<float> node_costs = node_weight_sums<VectorType>(graph, false);
    VectorType<float> positive_node_costs = node_weight_sums<VectorType>(graph, true);
    VectorType<int> flags = edge_criterion_flags<VectorType>(graph, node_costs);
    VectorType<int> triangle_edges = triangle_criterion_edge_indices<VectorType>(
        graph,
        node_costs,
        positive_node_costs,
        opts.verbose);

    if (!triangle_edges.empty())
    {
        thrust::scatter(
            thrust::make_constant_iterator<int>(1),
            thrust::make_constant_iterator<int>(1) + triangle_edges.size(),
            triangle_edges.begin(),
            flags.begin());
    }

    if (opts.verbose)
        std::cout << "Number of edges found by persistency criteria: "
                  << thrust::reduce(flags.begin(), flags.end()) << "\n";

    return flags;
}

template<template<typename> class VectorType>
VectorType<int> identity_mapping(const int num_nodes)
{
    VectorType<int> mapping(num_nodes);
    thrust::sequence(mapping.begin(), mapping.end());
    return mapping;
}

template<template<typename> class VectorType>
void compose_node_mapping(const VectorType<int>& current_mapping, VectorType<int>& node_mapping)
{
    const int* const current_ptr = thrust::raw_pointer_cast(current_mapping.data());
    int* const node_mapping_ptr = thrust::raw_pointer_cast(node_mapping.data());
    const int current_size = current_mapping.size();

    thrust::for_each(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>((int)node_mapping.size()),
        [current_ptr, node_mapping_ptr, current_size] PREPROCESSOR_HOST_DEVICE (const int node) {
            const int mapped = node_mapping_ptr[node];
            if (mapped < current_size)
                node_mapping_ptr[node] = current_ptr[mapped];
        });
}

template<template<typename> class VectorType>
std::tuple<VectorType<int>, int> contraction_mapping(
    const Graph<VectorType>& graph,
    const multicut_solver_options& opts)
{
    VectorType<int> flags = contracting_edge_flags<VectorType>(graph, opts);
    const int persistent_edges = thrust::reduce(flags.begin(), flags.end());
    if (opts.verbose)
        std::cout << "Percentage of edges that are persistent: "
                  << (100.0f * (float)persistent_edges / (float)graph.num_directed_edges()) << "%\n";

    if (persistent_edges == 0
        || (float)persistent_edges < (float)graph.num_directed_edges() * opts.preprocessor_threshold)
    {
        return {identity_mapping<VectorType>(graph.num_nodes()), 0};
    }

    VectorType<int> tails(persistent_edges);
    VectorType<int> heads(persistent_edges);
    auto input = thrust::make_zip_iterator(
        thrust::make_tuple(graph.get_tails().begin(), graph.get_heads().begin()));
    auto output = thrust::make_zip_iterator(thrust::make_tuple(tails.begin(), heads.begin()));
    thrust::copy_if(input, input + flags.size(), flags.begin(), output, marked_edge());

    VectorType<int> components = connected_components::compute_cc<VectorType>(
        graph.num_nodes(),
        tails,
        heads);
    VectorType<int> mapping = compress_label_sequence<VectorType>(
        components,
        components.size() - 1);
    return {std::move(mapping), persistent_edges / 2};
}

} // namespace persistency_preprocessor_detail

template<template<typename> class VectorType>
VectorType<int> persistency_preprocess(Graph<VectorType>& graph, const multicut_solver_options& opts, const int iterations)
{
    VectorType<int> node_mapping = persistency_preprocessor_detail::identity_mapping<VectorType>(graph.num_nodes());
    const int limit = iterations < 0 ? persistency_preprocessor_detail::max_iterations : iterations;

    for (int iteration = 0; iteration < limit; ++iteration)
    {
        auto [current_mapping, persistent_edges] =
            persistency_preprocessor_detail::contraction_mapping<VectorType>(graph, opts);
        if (persistent_edges == 0)
            return node_mapping;

        Graph<VectorType> contracted = graph.contract(current_mapping);
        assert(contracted.num_nodes() < graph.num_nodes());
        contracted.remove_self_loops();
        persistency_preprocessor_detail::compose_node_mapping<VectorType>(current_mapping, node_mapping);
        graph = std::move(contracted);
    }

    return node_mapping;
}
