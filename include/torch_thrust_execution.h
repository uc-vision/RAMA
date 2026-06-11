#pragma once

#ifdef RAMA_TORCH_ALLOCATOR
#include <cstddef>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <thrust/system/cuda/execution_policy.h>

namespace rama {

inline cudaStream_t torch_current_cuda_stream()
{
    return at::cuda::getCurrentCUDAStream().stream();
}

struct torch_thrust_allocator {
    using value_type = char;

    char* allocate(std::ptrdiff_t bytes)
    {
        return static_cast<char*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
            static_cast<std::size_t>(bytes), torch_current_cuda_stream()));
    }

    void deallocate(char* pointer, std::size_t)
    {
        c10::cuda::CUDACachingAllocator::raw_delete(pointer);
    }
};

inline torch_thrust_allocator& torch_thrust_allocator_instance()
{
    static torch_thrust_allocator allocator;
    return allocator;
}

inline auto torch_thrust_execution()
{
    return thrust::cuda::par(torch_thrust_allocator_instance()).on(torch_current_cuda_stream());
}

} // namespace rama

#define RAMA_THRUST_EXEC ::rama::torch_thrust_execution(),
#else
#define RAMA_THRUST_EXEC
#endif
