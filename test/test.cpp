#include "pch.h"
#include "../block_convolution/fft_cpu.h"
#include "../block_convolution/overlap_conv.hpp"

#define EXPECT_NEQ(a, b) EXPECT_TRUE(neq(a, b))

template <typename T>
bool neq(const T& a, const T& b, double eps = 0.000001)
{ return std::abs(a - b) < eps; }

namespace conv {

  TEST(fft, fft_recursive) {
    std::vector<comp_t> input = { {1, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {2, 0}, {5, 0}, {4,0} };
    // F_ans is obtained by numpy.fft.fft
    std::vector<double> F_real_ans = {
      14.f,
      2.414213562f,
      -6.f,
      -0.41421356f,
      2.f,
      -0.41421356f,
      -6.f,
      2.414213562f,
    };

    std::vector<double> F_imag_ans = {
      0.f,
      7.242640687f,
      2.f,
      1.242640687f,
      0.f,
      -1.242640687f,
      -2.f,
      -7.242640687f,
    };

    conv::fft_recursive(input);

    // check fft answer
    for (int i = 0; i < input.size(); i++) {
      EXPECT_NEQ(input[i].real(), F_real_ans[i]);
      EXPECT_NEQ(input[i].imag(), F_imag_ans[i]);
    }
  }

  TEST(fft, ifft_recursive) {
    std::vector<comp_t> input = { {1, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {2, 0}, {5, 0}, {4, 0} };
    auto input_copy = input;

    conv::fft_recursive(input);
    conv::ifft_recursive(input);

    // check ifft answer
    for (int i = 0; i < input.size(); i++) {
      EXPECT_NEQ(input[i].real(), input_copy[i].real());
      EXPECT_NEQ(input[i].imag(), input_copy[i].imag());
    }
  }

  TEST(fft, fft_non_recursive) {
    std::vector<comp_t> input = { {1, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {2, 0}, {5, 0}, {4,0} };
    auto input_copy = input;

    conv::fft(input);
    conv::fft_recursive(input_copy);

    // check fft answer
    for (int i = 0; i < input.size(); i++) {
      EXPECT_NEQ(input[i].real(), input_copy[i].real());
      EXPECT_EQ(input[i].imag(), input_copy[i].imag());
    }
  }

  TEST(fft, fft_stockham) {
    std::vector<comp_t> input = { {1, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {2, 0}, {5, 0}, {4,0} };
    auto input_copy = input;

    conv::fft_stockham(input);
    conv::fft_recursive(input_copy);

    // check fft answer
    for (int i = 0; i < input.size(); i++) {
      EXPECT_NEQ(input[i].real(), input_copy[i].real());
      EXPECT_EQ(input[i].imag(), input_copy[i].imag());
    }
  }

  TEST(fft, fft_stockham_for) {
    std::vector<comp_t> input = { {1, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {2, 0}, {5, 0}, {4,0} };
    auto input_copy = input;

    auto res = conv::fft_stockham_for(input);
    conv::fft_recursive(input_copy);

    // check fft answer
    for (int i = 0; i < input.size(); i++) {
      EXPECT_NEAR(res[i].real(), input_copy[i].real(), 0.00001);
      EXPECT_EQ(res[i].imag(), input_copy[i].imag());
    }
  }

  TEST(overlap, save) {
    std::vector<comp_t> data = { {3, 0}, {-1, 0}, {0, 0}, {1, 0}, {3, 0}, {2, 0}, {0, 0}, {1, 0}, {2, 0}, {1, 0} };
    std::vector<comp_t> filter = { {1, 0}, {1, 0}, {1, 0} };
    std::vector<comp_t> ans = {
      {3, 0}, {2, 0}, {2, 0}, {0, 0}, {4, 0}, {6, 0}, {5, 0}, {3, 0}, {3, 0}, {4, 0}, {3, 0}, {1, 0}
    };

    auto ret = overlap_save(
      data,
      filter,
      4);

    for (int i = 0; i < ans.size(); i++) {
      EXPECT_NEAR(ret[i].real(), ans[i].real(), 0.00001);
      EXPECT_NEAR(ret[i].imag(), ans[i].imag(), 0.00001);
    }
  }

  TEST(overlap, add) {
    std::vector<comp_t> data = { {3, 0}, {-1, 0}, {0, 0}, {1, 0}, {3, 0}, {2, 0}, {0, 0}, {1, 0}, {2, 0}, {1, 0} };
    std::vector<comp_t> filter = { {1, 0}, {1, 0}, {1, 0} };
    std::vector<comp_t> ans = {
      {3, 0}, {2, 0}, {2, 0}, {0, 0}, {4, 0}, {6, 0}, {5, 0}, {3, 0}, {3, 0}, {4, 0}, {3, 0}, {1, 0}
    };

    auto ret = overlap_add(
      data,
      filter,
      4);

    for (int i = 0; i < ans.size(); i++) {
      EXPECT_NEAR(ret[i].real(), ans[i].real(), 0.00001);
      EXPECT_NEAR(ret[i].imag(), ans[i].imag(), 0.00001);
    }
  }
} // namespace hnll::audio