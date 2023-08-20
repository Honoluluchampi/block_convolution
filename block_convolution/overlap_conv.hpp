#pragma once
#include "fft_cpu.h"

namespace conv {

std::vector<comp_t> overlap_save(
  const std::vector<comp_t>& data,
  std::vector<comp_t>& filter,
  int block_size) {

  std::vector<comp_t> ret;

  int n = block_size;
  int m = filter.size();
  int l = n - m + 1;

  // 0 padding for filter
  filter.resize(block_size);

  for (int i = 0; i <= data.size() / l; i++) {
    // extract block data
    std::vector<comp_t> block(n);
    for (int j = 0; j < n; j++) {
      int data_idx = j + i * l - (m - 1);
      // 0 padding for data
      if (data_idx < 0 || data_idx > data.size())
        block[j] = 0;
      else
        block[j] = data[data_idx];
    }

    // conv
    auto filter_copy = filter;
    fft_recursive(filter_copy);
    fft_recursive(block);

    for (int j = 0; j < n; j++) {
      block[j] *= filter_copy[j];
    }

    ifft_recursive(block);

    for (int j = 0; j < l; j++) {
      ret.emplace_back(block[m - 1 + j]);
    }
  }

  return ret;
}

} // namespace conv