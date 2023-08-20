// https://www.slideshare.net/GourabGhosh4/overlap-add-overlap-savedigital-signal-processing

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

std::vector<comp_t> overlap_add(
  std::vector<comp_t>& data,
  std::vector<comp_t>& filter,
  int block_size
  ) {

  int n = block_size;
  int m = filter.size();
  int l = n - m + 1;

  std::vector<comp_t> ret(data.size() + m - 1, 0);

  // zero padding
  filter.resize(filter.size() + l - 1);

  for (int i = 0; i < data.size() / l; i++) {
    // prepare block
    std::vector<comp_t> block(n, 0);
    for (int j = 0; j < l; j++) {
      block[j] = data[i * l + j];
    }

    auto filter_copy = filter;

    // conv
    fft_recursive(block);
    fft_recursive(filter_copy);

    for (int j = 0; j < block.size(); j++) {
      block[j] *= filter_copy[j];
    }

    ifft_recursive(block);

    // overlap add
    for (int j = 0; j < block.size(); j++) {
      ret[i * l + j] += block[j];
    }
  }

  return ret;
}

} // namespace conv