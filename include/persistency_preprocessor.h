#pragma once

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
                && ((node_costs_ptr[tail] - 2.0f * cost < 0.0f)
                    || (node_costs_ptr[head] - 2.0f * cost < 0.0f));
        });

    return flags;
}

template<template<typename> class VectorType>
std::tuple<VectorType<int>, VectorType<int>> candidate_edges_from_flags(
    const Graph<VectorType>& graph,
    const VectorType<int>& flags)
{
    const int candidate_count = thrust::reduce(flags.begin(), flags.end());
    VectorType<int> tails(candidate_count);
    VectorType<int> heads(candidate_count);

    auto input = thrust::make_zip_iterator(
        thrust::make_tuple(graph.get_tails().begin(), graph.get_heads().begin()));
    auto output = thrust::make_zip_iterator(thrust::make_tuple(tails.begin(), heads.begin()));
    thrust::copy_if(input, input + flags.size(), flags.begin(), output, marked_edge());

    auto edge_begin = thrust::make_zip_iterator(thrust::make_tuple(tails.begin(), heads.begin()));
    auto edge_end = thrust::make_zip_iterator(thrust::make_tuple(tails.end(), heads.end()));
    thrust::for_each(
        edge_begin,
        edge_end,
        [] PREPROCESSOR_HOST_DEVICE (const thrust::tuple<int&, int&>& edge) {
            int& tail = thrust::get<0>(edge);
            int& head = thrust::get<1>(edge);
            const int normalized_tail = thrust::min(tail, head);
            const int normalized_head = thrust::max(tail, head);
            tail = normalized_tail;
            head = normalized_head;
        });
    thrust::sort(edge_begin, edge_end);
    auto unique_end = thrust::unique(edge_begin, edge_end);
    const int unique_count = unique_end - edge_begin;
    tails.resize(unique_count);
    heads.resize(unique_count);
    return {std::move(tails), std::move(heads)};
}

template<template<typename> class VectorType>
VectorType<int> contracting_edge_flags(const Graph<VectorType>& graph, const multicut_solver_options& opts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;

    VectorType<float> node_costs = node_weight_sums<VectorType>(graph, false);
    VectorType<int> flags = edge_criterion_flags<VectorType>(graph, node_costs);

    if (opts.verbose)
        std::cout << "Number of directed edges found by safe persistency criterion: "
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
    if (persistent_edges == 0
        || (float)persistent_edges < (float)graph.num_directed_edges() * opts.preprocessor_threshold)
    {
        return {identity_mapping<VectorType>(graph.num_nodes()), 0};
    }

    auto [candidate_tails, candidate_heads] = candidate_edges_from_flags<VectorType>(graph, flags);
    const int candidate_count = candidate_tails.size();
    if (candidate_count == 0)
        return {identity_mapping<VectorType>(graph.num_nodes()), 0};

    VectorType<int> incident_nodes(2 * candidate_count);
    VectorType<int> incident_edges(2 * candidate_count);
    thrust::copy(candidate_tails.begin(), candidate_tails.end(), incident_nodes.begin());
    thrust::copy(candidate_heads.begin(), candidate_heads.end(), incident_nodes.begin() + candidate_count);
    thrust::copy(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(candidate_count), incident_edges.begin());
    thrust::copy(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(candidate_count), incident_edges.begin() + candidate_count);
    thrust::sort_by_key(incident_nodes.begin(), incident_nodes.end(), incident_edges.begin());

    VectorType<int> unique_nodes(incident_nodes.size());
    VectorType<int> best_indices(incident_edges.size());
    auto best_end = thrust::reduce_by_key(
        incident_nodes.begin(),
        incident_nodes.end(),
        incident_edges.begin(),
        unique_nodes.begin(),
        best_indices.begin(),
        thrust::equal_to<int>(),
        thrust::minimum<int>());
    const int unique_node_count = best_end.first - unique_nodes.begin();

    VectorType<int> best_edge(graph.num_nodes(), candidate_count);
    thrust::scatter(
        best_indices.begin(),
        best_indices.begin() + unique_node_count,
        unique_nodes.begin(),
        best_edge.begin());

    VectorType<int> selected(candidate_count);
    const int* const tails_ptr = thrust::raw_pointer_cast(candidate_tails.data());
    const int* const heads_ptr = thrust::raw_pointer_cast(candidate_heads.data());
    const int* const best_edge_ptr = thrust::raw_pointer_cast(best_edge.data());
    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(candidate_count),
        selected.begin(),
        [tails_ptr, heads_ptr, best_edge_ptr] PREPROCESSOR_HOST_DEVICE (const int edge) {
            return best_edge_ptr[tails_ptr[edge]] == edge && best_edge_ptr[heads_ptr[edge]] == edge;
        });

    const int selected_count = thrust::reduce(selected.begin(), selected.end());
    if (selected_count == 0)
        return {identity_mapping<VectorType>(graph.num_nodes()), 0};

    VectorType<int> selected_tails(selected_count);
    VectorType<int> selected_heads(selected_count);
    auto selected_input = thrust::make_zip_iterator(thrust::make_tuple(candidate_tails.begin(), candidate_heads.begin()));
    auto selected_output = thrust::make_zip_iterator(thrust::make_tuple(selected_tails.begin(), selected_heads.begin()));
    thrust::copy_if(selected_input, selected_input + selected.size(), selected.begin(), selected_output, marked_edge());

    VectorType<int> mapping = identity_mapping<VectorType>(graph.num_nodes());
    thrust::scatter(selected_tails.begin(), selected_tails.end(), selected_heads.begin(), mapping.begin());
    mapping = compress_label_sequence<VectorType>(mapping, mapping.size() - 1);

    if (opts.verbose)
        std::cout << "Safe preprocessor: candidates=" << candidate_count
                  << ", matched=" << selected_count << "\n";

    return {std::move(mapping), selected_count};
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
