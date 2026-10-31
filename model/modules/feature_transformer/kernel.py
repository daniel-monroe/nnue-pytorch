import cupy as cp
import torch


def _find_nearest_divisor(value: int, target: int) -> int:
    divisors = []
    for i in range(1, value + 1):
        if value % i == 0:
            divisors.append((i, abs(target - i)))
    divisors.sort(key=lambda x: x[1])
    return divisors[0][0]


_num_threads_forward_cache: dict[int, int] = dict()


def _get_num_threads_for_forward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_forward_cache:
        _num_threads_forward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_forward_cache[output_size]


_num_threads_backward_cache: dict[int, int] = dict()


def _get_num_threads_for_backward(output_size: int) -> int:
    optimal_num_threads = 512
    if output_size not in _num_threads_backward_cache:
        _num_threads_backward_cache[output_size] = _find_nearest_divisor(
            output_size, optimal_num_threads
        )

    return _num_threads_backward_cache[output_size]


def _kernel_with_threads(kernel, threads):
    def f(grid, args):
        kernel(grid=grid, block=threads, args=args)

    return f


_sparse_input_linear_forward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_forward_kernel(max_active_indices: int, output_size: int):
    """
    @param: max_active_indices
        The maximum number of indices that are non-zero
        for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @param: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.
    """
    num_threads = _get_num_threads_for_forward(output_size)
    output_thread_slice_size = output_size // num_threads
    key = (max_active_indices, output_size, num_threads)
    if key not in _sparse_input_linear_forward_kernel_cache:
        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: input_indices
        A matrix of shape (BATCH_SIZE, max_active_indices)
        containing indices of active indices for each position
        in a batch. Input index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the input_indices matrix.
        The type for input indices is int32_t.

    @param: input_values
        A matrix of shape (BATCH_SIZE, max_active_indices)
        containing the values (arity) of the corresponding
        input index in input_indices.
        The type for the input value (arity) is float32.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type float32.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type float32.
*/
void sparse_input_linear_forward(
    const int32_t* const input_indices,
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {{
    __shared__
          float          shared_output[{output_size}];

    const uint32_t       block_idx           = blockIdx.x;
    const uint32_t       slice_offset        = threadIdx.x * {output_thread_slice_size};

          float*   const output_slice        = output + block_idx * {output_size} + slice_offset;
    const float*   const bias_slice          = bias                               + slice_offset;
          float*         shared_output_slice = shared_output                      + slice_offset;

    const int32_t* const input_index_row     = input_indices + block_idx * {max_active_indices};

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        shared_output_slice[s] = bias_slice[s];
    }}

    for (uint32_t k = 0; k < {max_active_indices}; ++k)
    {{
        const int32_t input_index = input_index_row[k];
        if (input_index != -1)
        {{
            const float* const weight_slice = weight + input_index * {output_size} + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            {{
                shared_output_slice[s] += weight_slice[s];
            }}
        }} else break;
    }}

    #pragma unroll
    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
    {{
        output_slice[s] = shared_output_slice[s];
    }}
}}

""".format(
                max_active_indices=max_active_indices,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
            ),
            "sparse_input_linear_forward",
        )
        kernel.compile()
        _sparse_input_linear_forward_kernel_cache[key] = _kernel_with_threads(
            kernel, (num_threads,)
        )
    return _sparse_input_linear_forward_kernel_cache[key]


_sparse_input_linear_backward_kernel_cache = dict()


@torch.compiler.disable(recursive=False)
def make_sparse_input_linear_backward_kernel(max_active_indices: int, output_size: int):
    """
    Weight-gradient kernel that works on (feature, position) pairs sorted by
    feature instead of scattering every active feature with global atomics.

    Each block owns a fixed chunk of {chunk_size} consecutive sorted pairs and
    sums the output gradients of every run of equal features in registers.
    Runs that lie entirely inside the chunk are written with a plain store;
    only the (at most two) runs that cross a chunk boundary use atomicAdd.
    The weight gradient must be zero initialized, as rows of features that
    are not active anywhere in the batch are never touched.

    Kernel arguments:
        sorted_indices: int32[n], the flattened feature indices sorted
            ascending; padding entries (-1) come first and are skipped.
        sort_order: int64[n], positions of sorted_indices in the flattened
            input, so the batch row is sort_order // max_active_indices.
        weight_grad: float32[num_inputs, output_size], zero initialized.
        output_grad: float32[batch_size, output_size].
        n: uint32, number of pairs (batch_size * max_active_indices).

    The grid must be ceil(n / {chunk_size}) blocks.
    """
    num_threads = _get_num_threads_for_backward(output_size)
    output_thread_slice_size = output_size // num_threads
    chunk_size = 64
    key = (max_active_indices, output_size, num_threads, chunk_size)
    if key not in _sparse_input_linear_backward_kernel_cache:
        kernel = cp.RawKernel(
            r"""

typedef unsigned int uint32_t;
typedef int int32_t;
typedef long long int64_t;

extern "C" __global__
void sparse_input_linear_backward_sorted(
    const int32_t* const __restrict__ sorted_indices,
    const int64_t* const __restrict__ sort_order,
          float*   const __restrict__ weight_grad,
    const float*   const __restrict__ output_grad,
    const uint32_t                    n
) {{
    __shared__ int32_t  shared_index[{chunk_size}];
    __shared__ uint32_t shared_row[{chunk_size}];

    const uint32_t begin = blockIdx.x * {chunk_size};
    const uint32_t end   = min(begin + {chunk_size}, n);

    // Padding (-1) sorts first, so a chunk ending in padding has no work.
    if (sorted_indices[end - 1] < 0)
        return;

    for (uint32_t i = threadIdx.x; i < {chunk_size}; i += blockDim.x)
    {{
        const uint32_t j = begin + i;
        if (j < end)
        {{
            shared_index[i] = sorted_indices[j];
            shared_row[i]   = (uint32_t)(sort_order[j] / {max_active_indices});
        }}
        else
            shared_index[i] = -1;
    }}
    __syncthreads();

    // Features whose run continues from the previous or into the next chunk.
    const int32_t prev_index = begin > 0 ? sorted_indices[begin - 1] : -1;
    const int32_t next_index = end   < n ? sorted_indices[end]       : -1;

    float acc[{output_thread_slice_size}];
    int32_t current = -1;

    for (uint32_t i = 0; i < {chunk_size}; ++i)
    {{
        const int32_t index = shared_index[i];
        if (index < 0)
            continue;

        if (index != current)
        {{
            if (current >= 0)
            {{
                float* const row = weight_grad + (size_t)(unsigned)current * {output_size} + threadIdx.x;
                if (current == prev_index || current == next_index)
                {{
                    #pragma unroll
                    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
                        atomicAdd(&row[s * {num_threads}], acc[s]);
                }}
                else
                {{
                    #pragma unroll
                    for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
                        row[s * {num_threads}] = acc[s];
                }}
            }}
            current = index;
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
                acc[s] = 0.0f;
        }}

        const float* const grad_row = output_grad + (size_t)shared_row[i] * {output_size} + threadIdx.x;
        #pragma unroll
        for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
            acc[s] += grad_row[s * {num_threads}];
    }}

    if (current >= 0)
    {{
        float* const row = weight_grad + (size_t)(unsigned)current * {output_size} + threadIdx.x;
        if (current == prev_index || current == next_index)
        {{
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
                atomicAdd(&row[s * {num_threads}], acc[s]);
        }}
        else
        {{
            #pragma unroll
            for (uint32_t s = 0; s < {output_thread_slice_size}; ++s)
                row[s * {num_threads}] = acc[s];
        }}
    }}
}}

""".format(
                max_active_indices=max_active_indices,
                output_thread_slice_size=output_thread_slice_size,
                output_size=output_size,
                num_threads=num_threads,
                chunk_size=chunk_size,
            ),
            "sparse_input_linear_backward_sorted",
        )
        kernel.compile()
        _sparse_input_linear_backward_kernel_cache[key] = (
            _kernel_with_threads(kernel, (num_threads,)),
            chunk_size,
        )
    return _sparse_input_linear_backward_kernel_cache[key]
