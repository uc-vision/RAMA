#include <vector>
#include <tuple>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rama_solver.h"
#include "multicut_solver_options.h"
#include "multicut_text_parser.h"
#include "rama_utils.h"

#ifdef WITH_TORCH
#include <torch/extension.h>
#include <torch/torch.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <algorithm>
#include <type_traits>
#include <utility>
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename T>
torch::ScalarType torch_scalar_type()
{
    if constexpr (std::is_same_v<T, int>)
        return torch::kInt32;
    else if constexpr (std::is_same_v<T, long long>)
        return torch::kInt64;
    else if constexpr (std::is_same_v<T, float>)
        return torch::kFloat32;
    else
        static_assert(std::is_same_v<T, int> || std::is_same_v<T, long long> || std::is_same_v<T, float>,
                      "Unsupported torch_tensor_vector element type");
}

static thread_local int torch_tensor_vector_device = 0;

template <typename T>
class torch_tensor_vector {
public:
    using value_type = T;
    using pointer = thrust::device_ptr<T>;
    using const_pointer = thrust::device_ptr<const T>;
    using size_type = std::size_t;
    using iterator = pointer;
    using const_iterator = const_pointer;

    torch_tensor_vector()
        : tensor_(allocate_tensor(0))
    {}

    explicit torch_tensor_vector(const size_type count)
        : tensor_(allocate_tensor(count))
    {}

    torch_tensor_vector(const size_type count, const T value)
        : tensor_(allocate_tensor(count))
    {
        thrust::fill(begin(), end(), value);
    }

    template <typename Iterator>
    torch_tensor_vector(Iterator first, Iterator last)
        : tensor_(allocate_tensor(std::distance(first, last)))
    {
        thrust::copy(first, last, begin());
    }

    torch_tensor_vector(const torch_tensor_vector& other)
        : tensor_(allocate_tensor(other.size()))
    {
        thrust::copy(other.begin(), other.end(), begin());
    }

    torch_tensor_vector(torch_tensor_vector&& other) noexcept = default;

    torch_tensor_vector& operator=(const torch_tensor_vector& other)
    {
        if (this != &other)
        {
            tensor_ = allocate_tensor(other.size());
            thrust::copy(other.begin(), other.end(), begin());
        }
        return *this;
    }

    torch_tensor_vector& operator=(torch_tensor_vector&& other) noexcept = default;

    iterator begin()
    {
        return thrust::device_pointer_cast(tensor_.data_ptr<T>());
    }

    iterator end()
    {
        return begin() + size();
    }

    const_iterator begin() const
    {
        return thrust::device_pointer_cast(static_cast<const T*>(tensor_.data_ptr<T>()));
    }

    const_iterator end() const
    {
        return begin() + size();
    }

    pointer data()
    {
        return begin();
    }

    const_pointer data() const
    {
        return begin();
    }

    size_type size() const
    {
        return static_cast<size_type>(tensor_.numel());
    }

    bool empty() const
    {
        return size() == 0;
    }

    decltype(std::declval<pointer>()[0]) operator[](const size_type index)
    {
        return begin()[index];
    }

    decltype(std::declval<const_pointer>()[0]) operator[](const size_type index) const
    {
        return begin()[index];
    }

    decltype(std::declval<pointer>()[0]) back()
    {
        return begin()[size() - 1];
    }

    decltype(std::declval<const_pointer>()[0]) back() const
    {
        return begin()[size() - 1];
    }

    void resize(const size_type count)
    {
        resize(count, T{});
    }

    void resize(const size_type count, const T value)
    {
        if (count == size())
            return;

        torch::Tensor resized = allocate_tensor(count);
        const size_type copy_count = std::min(count, size());
        if (copy_count > 0)
            thrust::copy(begin(), begin() + copy_count, thrust::device_pointer_cast(resized.data_ptr<T>()));
        if (count > copy_count)
            thrust::fill(thrust::device_pointer_cast(resized.data_ptr<T>()) + copy_count,
                         thrust::device_pointer_cast(resized.data_ptr<T>()) + count, value);
        tensor_ = resized;
    }

    void swap(torch_tensor_vector& other) noexcept
    {
        std::swap(tensor_, other.tensor_);
    }

    torch::Tensor tensor() const
    {
        return tensor_;
    }

private:
    static torch::Tensor allocate_tensor(const size_type count)
    {
        return torch::empty(
            {static_cast<long>(count)},
            torch::TensorOptions()
                .device(torch::Device(torch::kCUDA, torch_tensor_vector_device))
                .dtype(torch_scalar_type<T>()));
    }

    torch::Tensor tensor_;
};

template <typename T>
void swap(torch_tensor_vector<T>& left, torch_tensor_vector<T>& right) noexcept
{
    left.swap(right);
}
#endif

namespace py=pybind11;

static std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
solve_gpu(thrust::device_vector<int>&& i_gpu, thrust::device_vector<int>&& j_gpu,
          thrust::device_vector<float>&& costs_gpu, const multicut_solver_options& opts)
{
    thrust::device_vector<int> sanitized_node_ids;
    if (opts.sanitize_graph)
        sanitized_node_ids = compute_sanitized_graph(i_gpu, j_gpu, costs_gpu);
    else
        sort_edge_nodes<thrust::device_vector>(i_gpu, j_gpu);

    DeviceGraph G(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu));
    auto [node_mapping, lb, timeline] = rama_solver<thrust::device_vector>(G, opts);

    if (opts.sanitize_graph)
        node_mapping = desanitize_node_labels(node_mapping, sanitized_node_ids);

    return {std::move(node_mapping), lb, std::move(timeline)};
}

#ifdef WITH_TORCH
static std::tuple<torch_tensor_vector<int>, double, std::vector<std::vector<int>>>
solve_torch_gpu(torch_tensor_vector<int>&& i_gpu, torch_tensor_vector<int>&& j_gpu,
          torch_tensor_vector<float>&& costs_gpu, const multicut_solver_options& opts)
{
    if (opts.sanitize_graph)
        throw std::runtime_error("sanitize_graph is not supported by rama_torch");

    sort_edge_nodes<torch_tensor_vector>(i_gpu, j_gpu);
    Graph<torch_tensor_vector> G(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu));
    return rama_solver<torch_tensor_vector>(G, opts);
}
#endif

static std::tuple<std::vector<int>, double, int, std::vector<std::vector<int>>>
rama_cuda(const std::vector<int>& i, const std::vector<int>& j,
          const std::vector<float>& costs, const multicut_solver_options& opts)
{
    initialize_gpu(opts.verbose);
    thrust::device_vector<int> i_gpu(i.begin(), i.end());
    thrust::device_vector<int> j_gpu(j.begin(), j.end());
    thrust::device_vector<float> costs_gpu(costs.begin(), costs.end());

    auto start = std::chrono::steady_clock::now();
    auto [node_mapping, lb, timeline] = solve_gpu(std::move(i_gpu), std::move(j_gpu),
                                                   std::move(costs_gpu), opts);
    auto end = std::chrono::steady_clock::now();
    int dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::vector<int> h_node_mapping(node_mapping.size());
    thrust::copy(node_mapping.begin(), node_mapping.end(), h_node_mapping.begin());
    return {h_node_mapping, lb, dur, timeline};
}

#ifdef WITH_TORCH
std::vector<torch::Tensor> rama_torch(
    const torch::Tensor& _i,
    const torch::Tensor& _j,
    const torch::Tensor& _costs,
    const multicut_solver_options& opts)
{
    CHECK_INPUT(_i);
    CHECK_INPUT(_j);
    CHECK_INPUT(_costs);
    if (_i.size(0) != _j.size(0) || _i.size(0) != _costs.size(0))
        throw std::runtime_error("Input shapes must match");
    if (_i.scalar_type() != _j.scalar_type())
        throw std::runtime_error("Node indices i, j should be of same type");
    if (_i.scalar_type() != torch::kInt32)
        throw std::runtime_error("Node indices i, j should be int32");
    if (_costs.scalar_type() != torch::kFloat32)
        throw std::runtime_error("costs should be float32");

    TORCH_CHECK(_i.dim() == 1, "i should be one-dimensional");
    TORCH_CHECK(_j.dim() == 1, "j should be one-dimensional");
    TORCH_CHECK(_costs.dim() == 1, "costs should be one-dimensional");

    const int device = _costs.device().index();
    if (device < 0)
        throw std::runtime_error("Invalid device ID");
    cudaSetDevice(device);
    torch_tensor_vector_device = device;

    auto i_begin = thrust::device_pointer_cast(_i.data_ptr<int32_t>());
    auto j_begin = thrust::device_pointer_cast(_j.data_ptr<int32_t>());
    auto costs_begin = thrust::device_pointer_cast(_costs.data_ptr<float>());
    torch_tensor_vector<int> i(i_begin, i_begin + _i.size(0));
    torch_tensor_vector<int> j(j_begin, j_begin + _j.size(0));
    torch_tensor_vector<float> costs(costs_begin, costs_begin + _costs.size(0));

    auto [node_mapping, lb, timeline] = solve_torch_gpu(std::move(i), std::move(j), std::move(costs), opts);

    torch::Tensor node_mapping_torch = node_mapping.tensor();
    torch::Tensor lb_torch = at::empty({1}, _costs.options().dtype(torch::kFloat64));
    lb_torch.fill_(lb);
    return {node_mapping_torch, lb_torch};
}
#endif

std::vector<std::vector<int>> rama_cuda_gpu_pointers(const int* const i, const int* const j, const float* const edge_costs,
                        int* const node_labels, const int num_nodes, const int num_edges, const int gpuDeviceID, const multicut_solver_options& opts)
{
    cudaSetDevice(gpuDeviceID);
    thrust::device_vector<int> i_thrust(i, i + num_edges);
    thrust::device_vector<int> j_thrust(j, j + num_edges);
    thrust::device_vector<float> costs_thrust(edge_costs, edge_costs + num_edges);

    auto [node_mapping, lb, timeline] = solve_gpu(std::move(i_thrust), std::move(j_thrust),
                                                   std::move(costs_thrust), opts);
    thrust::copy(node_mapping.begin(), node_mapping.end(), node_labels);
    return timeline;
}

PYBIND11_MODULE(rama_py, m) {
    m.doc() = "Bindings for RAMA: Rapid algorithm for multicut. "
                "For running purely primal algorithm initialize multicut_solver_options with \"P\". "
                "For algorithm with best quality call with \"PD+\" where \"PD\" is default algorithm. "
                "For only computing the lower bound call with \"D\". ";
    py::class_<multicut_solver_options>(m, "multicut_solver_options")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def(py::init<const int&, const int&, const int&, const int&, const int&,
                const float&, const float&, const float&,
                const bool&, const int&, const bool&>())
        .def_readwrite("max_cycle_length_lb", &multicut_solver_options::max_cycle_length_lb)
        .def_readwrite("num_dual_itr_lb", &multicut_solver_options::num_dual_itr_lb)
        .def_readwrite("max_cycle_length_primal", &multicut_solver_options::max_cycle_length_primal)
        .def_readwrite("num_dual_itr_primal", &multicut_solver_options::num_dual_itr_primal)
        .def_readwrite("num_outer_itr_dual", &multicut_solver_options::num_outer_itr_dual)
        .def_readwrite("mean_multiplier_mm", &multicut_solver_options::mean_multiplier_mm)
        .def_readwrite("matching_thresh_crossover_ratio", &multicut_solver_options::matching_thresh_crossover_ratio)
        .def_readwrite("tri_memory_factor", &multicut_solver_options::tri_memory_factor)
        .def_readwrite("only_compute_lb", &multicut_solver_options::only_compute_lb)
        .def_readwrite("max_time_sec", &multicut_solver_options::max_time_sec)
        .def_readwrite("verbose", &multicut_solver_options::verbose)
        .def_readwrite("dump_timeline", &multicut_solver_options::dump_timeline)
        .def_readwrite("sanitize_graph", &multicut_solver_options::sanitize_graph)
        .def("__repr__", [](const multicut_solver_options &a) {
            return a.get_string();
        });

    m.def("rama_cuda", [](const std::vector<int>& i, const std::vector<int>& j, const std::vector<float>& edge_costs, const multicut_solver_options& opts) {
            return rama_cuda(i, j, edge_costs, opts);
            });

    m.def("rama_cuda_gpu_pointers", [](const long i_ptr, const long j_ptr, const long edge_costs_ptr, const long node_labels_out_ptr,
                                    const int num_nodes, const int num_edges, const int gpuDeviceID, const multicut_solver_options& opts) {
            const int* const i = reinterpret_cast<const int* const>(i_ptr);
            const int* const j = reinterpret_cast<const int* const>(j_ptr);
            const float* const edge_costs = reinterpret_cast<const float* const>(edge_costs_ptr);
            int* const node_labels = reinterpret_cast<int* const>(node_labels_out_ptr);
            return rama_cuda_gpu_pointers(i, j, edge_costs, node_labels, num_nodes, num_edges, gpuDeviceID, opts);
            });

    m.def("read_multicut_file", [](const std::string& filename) {
            return read_file(filename);
            });

    #ifdef WITH_TORCH
        m.def("rama_torch", &rama_torch, "RAMA CUDA solver with torch interface.");
    #endif
}
