#include "fft_cpu.h"

// std
#include <iostream>

// cuda
#include <cuComplex.h>

__device__ int ID2(int r, int c, int c_num) {
  return r * c_num + c;
}

__global__ void butterfly(
  cuDoubleComplex* src, 
  cuDoubleComplex* dst,
  int n,
  int item_per_thread,
  int a,
  int b) {
    int next_a = a << 1;
    int first_target = item_per_thread * threadIdx.x;

    for (int i = 0; i < item_per_thread; i++) {
      int target_id = first_target + i;
      int j = target_id / next_a;
      int k = target_id % next_a;
      int original_k = target_id % a;
      cuDoubleComplex w = make_cuDoubleComplex(
        cos(2 * M_PI / n * original_k * b),
        -sin(2 * M_PI / n * original_k * b)
      );
      if (k < a)
        dst[target_id] = cuCadd(src[ID2(j, original_k, a)], cuCmul(src[ID2(j + b, original_k, a)], w));
      else 
        dst[target_id] = cuCsub(src[ID2(j, original_k, a)], cuCmul(src[ID2(j + b, original_k, a)], w));
    }
}

auto fft_stockham_gpu(std::vector<comp_t>& input, int thread_count) {
  
  int n = input.size();

  cuDoubleComplex* x; // initialized by input
  cuDoubleComplex* y; // 0 init

  cudaMallocManaged(&x, n * sizeof(cuDoubleComplex));
  cudaMallocManaged(&y, n * sizeof(cuDoubleComplex));
  for (int i = 0; i < n; i++) {
    x[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
  }

  int p = std::log2(n);
  int a = 1;
  int b = n >> 1;

  // adjust item count per thread
  thread_count = std::min(thread_count, n);
  int item_per_thread = n / thread_count;

  for (int l = 0; l < p; l++) {
    if (l % 2 == 0)
      butterfly<<<1, thread_count>>>(x, y, n, item_per_thread, a, b);
    else
      butterfly<<<1, thread_count>>>(y, x, n, item_per_thread, a, b);

    cudaDeviceSynchronize();
    a <<= 1;
    b >>= 1;
  }

  std::vector<comp_t> output(n);
  for (int i = 0; i < n; i++) {
    if (p % 2 == 0)
      output[i] = { x[i].x, x[i].y };
    else
      output[i] = { y[i].x, y[i].y };
  }

  cudaFree(x);
  cudaFree(y);

  return output;
}

int main() {
  std::vector<comp_t> input = { {1, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {2, 0}, {5, 0}, {4,0} };
  auto input_copy = input;

  auto cpu_ret = conv::fft_stockham_for(input_copy);

  auto test_error = [&cpu_ret](std::vector<comp_t>& gpu_ret) {
    double max_error = 0.f;
    for (int i = 0; i < cpu_ret.size(); i++) {
      max_error = std::max(max_error, std::abs(gpu_ret[i] - cpu_ret[i]));
    }
    return max_error;
  };

  auto ret = fft_stockham_gpu(input, 1);
  std::cout << "thread count : 1 " << std::endl;
  std::cout << "\tmax error : " <<  test_error(ret) << std::endl;

  ret = fft_stockham_gpu(input, 4);
  std::cout << "thread count : 4 " << std::endl;
  std::cout << "\tmax error : " <<  test_error(ret) << std::endl;

  ret = fft_stockham_gpu(input, 8);
  std::cout << "thread count : 8 " << std::endl;
  std::cout << "\tmax error : " <<  test_error(ret) << std::endl;

  ret = fft_stockham_gpu(input, 16);
  std::cout << "thread count : 16 " << std::endl;
  std::cout << "\tmax error : " <<  test_error(ret) << std::endl;
}