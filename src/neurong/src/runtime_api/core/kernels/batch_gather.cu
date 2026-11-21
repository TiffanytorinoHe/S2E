#include <cuda_runtime.h>

// Gather from a list of device pointers into a contiguous array.
// This is the reverse of batch_copy_doubles_kernel in batch_copy.cu.
__global__ void batch_gather_floats_kernel(double** gpu_ptrs, float* out, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = static_cast<float>(*gpu_ptrs[idx]);
    }
}

extern "C" void launch_batch_gather_floats(double** d_gpu_ptrs, float* d_out, int count) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    batch_gather_floats_kernel<<<grid_size, block_size>>>(d_gpu_ptrs, d_out, count);
}

__global__ void response_presence_accumulate_kernel(double** voltage_ptrs,
                                                    double* probability_sum,
                                                    int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const double x = fmax(-60.0, fmin(60.0, *voltage_ptrs[idx] / 5.0));
        probability_sum[idx] += 1.0 / (1.0 + exp(-x));
    }
}

__global__ void response_presence_finalize_kernel(const double* probability_sum,
                                                  double* output,
                                                  int count,
                                                  double dt_ms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const double soft = probability_sum[idx] * dt_ms / 1.2;
        output[idx] = 1.0 - exp(-soft);
    }
}

extern "C" void launch_response_presence_accumulate(double** voltage_ptrs,
                                                     double* probability_sum,
                                                     int count) {
    const int block_size = 256;
    response_presence_accumulate_kernel<<<(count + block_size - 1) / block_size, block_size>>>(
        voltage_ptrs, probability_sum, count);
}

extern "C" void launch_response_presence_finalize(const double* probability_sum,
                                                   double* output,
                                                   int count,
                                                   double dt_ms) {
    const int block_size = 256;
    response_presence_finalize_kernel<<<(count + block_size - 1) / block_size, block_size>>>(
        probability_sum, output, count, dt_ms);
}
