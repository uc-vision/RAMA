#include <algorithm>
#include <chrono>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/distance.h>
#include <thrust/fill.h>

#include "multicut_solver_options.h"
#include "multicut_text_parser.h"
#include "rama_solver.h"
#include "rama_utils.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace py = pybind11;

namespace {

template <typename T>
torch::ScalarType torch_scalar_type()
{
    if constexpr (std::is_same_v<T, int>)
        return torch::kInt32;
    else if constexpr (std::is_same_v<T, std::int64_t>)
        return torch::kInt64;
    else if constexpr (std::is_same_v<T, float>)
        return torch::kFloat32;
    else
        static_assert(std::is_same_v<T, int> || std::is_same_v<T, std::int64_t> || std::is_same_v<T, float>,
                      "Unsupported torch_tensor_vector element type");
}

thread_local int torch_tensor_vector_device = 0;

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
        : tensor_(allocate_empty(0))
    {}

    explicit torch_tensor_vector(const size_type count)
        : tensor_(allocate_empty(count))
    {}

    torch_tensor_vector(const size_type count, const T value)
        : tensor_(allocate_full(count, value))
    {}

    explicit torch_tensor_vector(torch::Tensor tensor)
        : tensor_(std::move(tensor))
    {}

    template <typename Iterator, typename = std::enable_if_t<!std::is_integral_v<Iterator>>>
    torch_tensor_vector(Iterator first, Iterator last)
        : tensor_(allocate_empty(static_cast<size_type>(thrust::distance(first, last))))
    {
        thrust::copy(first, last, begin());
    }

    torch_tensor_vector(const torch_tensor_vector& other)
        : tensor_(other.tensor_.clone())
    {}

    __host__ torch_tensor_vector(torch_tensor_vector&& other) noexcept
        : tensor_(std::move(other.tensor_))
    {}

    torch_tensor_vector& operator=(const torch_tensor_vector& other)
    {
        if (this != &other)
            tensor_ = other.tensor_.clone();
        return *this;
    }

    __host__ torch_tensor_vector& operator=(torch_tensor_vector&& other) noexcept
    {
        tensor_ = std::move(other.tensor_);
        return *this;
    }

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

        torch::Tensor resized = allocate_empty(count);
        const size_type copy_count = std::min(count, size());
        if (copy_count > 0)
        {
            resized.slice(0, 0, static_cast<long>(copy_count))
                .copy_(tensor_.slice(0, 0, static_cast<long>(copy_count)));
        }
        if (count > copy_count)
        {
            resized.slice(0, static_cast<long>(copy_count), static_cast<long>(count)).fill_(value);
        }
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
    static torch::TensorOptions options()
    {
        return torch::TensorOptions()
            .device(torch::Device(torch::kCUDA, torch_tensor_vector_device))
            .dtype(torch_scalar_type<T>());
    }

    static torch::Tensor allocate_empty(const size_type count)
    {
        return torch::empty({static_cast<long>(count)}, options());
    }

    static torch::Tensor allocate_full(const size_type count, const T value)
    {
        return torch::full({static_cast<long>(count)}, value, options());
    }

    torch::Tensor tensor_;
};

template <typename T>
void swap(torch_tensor_vector<T>& left, torch_tensor_vector<T>& right) noexcept
{
    left.swap(right);
}

std::tuple<torch_tensor_vector<int>, double, std::vector<std::vector<int>>>
solve_torch_gpu(torch_tensor_vector<int>&& i_gpu,
                torch_tensor_vector<int>&& j_gpu,
                torch_tensor_vector<float>&& costs_gpu,
                const multicut_solver_options& opts)
{
    if (opts.sanitize_graph)
        throw std::runtime_error("sanitize_graph is not supported by rama_torch");

    sort_edge_nodes<torch_tensor_vector>(i_gpu, j_gpu);
    Graph<torch_tensor_vector> graph(std::move(i_gpu), std::move(j_gpu), std::move(costs_gpu));
    return rama_solver<torch_tensor_vector>(graph, opts);
}

std::vector<torch::Tensor> rama_torch(
    const torch::Tensor& input_i,
    const torch::Tensor& input_j,
    const torch::Tensor& input_costs,
    const multicut_solver_options& opts)
{
    CHECK_INPUT(input_i);
    CHECK_INPUT(input_j);
    CHECK_INPUT(input_costs);
    if (input_i.size(0) != input_j.size(0) || input_i.size(0) != input_costs.size(0))
        throw std::runtime_error("Input shapes must match");
    if (input_i.scalar_type() != input_j.scalar_type())
        throw std::runtime_error("Node indices i, j should be of same type");
    if (input_i.scalar_type() != torch::kInt32)
        throw std::runtime_error("Node indices i, j should be int32");
    if (input_costs.scalar_type() != torch::kFloat32)
        throw std::runtime_error("costs should be float32");

    TORCH_CHECK(input_i.dim() == 1, "i should be one-dimensional");
    TORCH_CHECK(input_j.dim() == 1, "j should be one-dimensional");
    TORCH_CHECK(input_costs.dim() == 1, "costs should be one-dimensional");

    const int device = input_costs.device().index();
    if (device < 0)
        throw std::runtime_error("Invalid device ID");
    cudaSetDevice(device);
    torch_tensor_vector_device = device;

    torch_tensor_vector<int> i(input_i.clone());
    torch_tensor_vector<int> j(input_j.clone());
    torch_tensor_vector<float> costs(input_costs.clone());

    auto [node_mapping, lower_bound, timeline] = solve_torch_gpu(std::move(i), std::move(j), std::move(costs), opts);

    torch::Tensor lower_bound_tensor = torch::empty({1}, input_costs.options().dtype(torch::kFloat64));
    lower_bound_tensor.fill_(lower_bound);
    return {node_mapping.tensor(), lower_bound_tensor};
}

} // namespace

PYBIND11_MODULE(rama_py, m) {
    m.doc() = "Bindings for RAMA: Rapid algorithm for multicut. "
              "For running purely primal algorithm initialize multicut_solver_options with \"P\". "
              "For algorithm with best quality call with \"PD+\" where \"PD\" is default algorithm. "
              "For only computing the lower bound call with \"D\". ";

    py::class_<multicut_solver_options>(m, "multicut_solver_options")
        .def(py::init<>())
        .def(py::init<const std::string&>())
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
        .def("__repr__", [](const multicut_solver_options& options) {
            return options.get_string();
        });

    m.def("rama_torch", &rama_torch, "RAMA CUDA solver with torch tensor storage.");
    m.def("read_multicut_file", [](const std::string& filename) {
        return read_file(filename);
    });
}
