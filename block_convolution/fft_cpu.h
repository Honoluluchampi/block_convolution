#pragma once

#include <cassert>
#include <vector>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>

using num_t = double;
using comp_t = std::complex<num_t>;

namespace conv {
	
	// all signals are represented as std::vector<comp>

	void zero_pad(std::vector<comp_t>& signal, size_t size) {
		auto original_size = signal.size();
		signal.resize(size);

		// if the target size is larger than the original size
		if (original_size < size) {
			for (int i = original_size; i < size; i++) {
				signal[i] = { 0.f, 0.f };
			}
		}
	}

	// currently Cooly-Tukey FFT
	void fft_rec(
		std::vector<comp_t>& input,
		int stride,
		int first_index,
		int bit,
		int n) {
		if (n > 1) {
			assert(n % 2 == 0 && "fft : input size must be equal to 2^k.");
			const int h = n / 2;
			const num_t theta = 2.f * M_PI / static_cast<num_t>(n);

			// butterflies
			for (int i = 0; i < h; i++) {
				const comp_t wi = { std::cos(i * theta), -sin(i * theta) };
				const comp_t a = input[first_index + i + 0];
				const comp_t b = input[first_index + i + h];

				input[first_index + i + 0] = a + b;
				input[first_index + i + h] = (a - b) * wi;
			}

			fft_rec(input, 2 * stride, first_index + 0, bit + 0,      h);
			fft_rec(input, 2 * stride, first_index + h, bit + stride, h);
		}
		// bit reverse
		else if (first_index > bit) {
			std::swap(input[first_index], input[bit]);
		}
	}

	// fft from time to freq
	void fft_recursive(std::vector<comp_t>& input) {
		int n = input.size();
		fft_rec(input, 1, 0, 0, n);
	}

	// ifft from freq to time
	void ifft_recursive(std::vector<comp_t>& input) {
		int n = input.size();

		for (int i = 0; i < n; i++) {
			input[i] = conj(input[i]);
		}

		fft_rec(input, 1, 0, 0, n);

		for (int i = 0; i < n; i++) {
			input[i] = conj(input[i] / static_cast<num_t>(n));
		}
	}

	// fft without recursion
	void fft(std::vector<comp_t>& input) {
		auto n = input.size();
		int h;
		num_t theta = 2 * M_PI / static_cast<num_t>(n);

		for (int m = n; (h = m >> 1) >= 1; m = h) {
			for (int i = 0; i < h; i++) {
				comp_t w = { std::cos(theta * i), -std::sin(theta * i) };
				for (int j = i; j < n; j += m) {
					// butterfly
					int k = j + h;
					comp_t tmp = input[j] - input[k];
					input[j] += input[k];
					input[k] = tmp * w;
				}
			}
			theta *= 2;
		}

		// in-place
		int i = 0;
		for (int j = 1; j < n - 1; j++) {
			for (int k = n >> 1; k > (i ^= k); k >>= 1);
			if (j < i) {
				std::swap(input[i], input[j]);
			}
		}
	}

	void fft_stockham_rec(int n, int s, bool flag, std::vector<comp_t>& input, std::vector<comp_t>& buffer) {
		const int m = n / 2;
		const double theta = 2 * M_PI / n;

		if (n == 1) {
			if (flag) {
				for (int q = 0; q < s; q++) {
					buffer[q] = input[q];
				}
			}
		}

		else {
			for (int p = 0; p < m; p++) {
				const comp_t w = comp_t{ cos(p * theta), -sin(p * theta) };
				for (int q = 0; q < s; q++) {
					const comp_t a = input[q + s * (p + 0)];
					const comp_t b = input[q + s * (p + m)];
					buffer[q + s * (2 * p + 0)] = a + b;
					buffer[q + s * (2 * p + 1)] = (a - b) * w;
				}
			}
			fft_stockham_rec(m, 2 * s, !flag, buffer, input);
		}
	}

	void fft_stockham(std::vector<comp_t>& input) {
		auto n = input.size();
		std::vector<comp_t> buffer(n);

		fft_stockham_rec(n, 1, 0, input, buffer);
	}

  inline int id2(int j, int k, int x_num)
  { return j * x_num + k; }

  void stockham_butterfly(
    const std::vector<comp_t>& src,
    std::vector<comp_t>& dst,
    int j,
    int k,
    int a,
    int b,
    int n) {
    comp_t w = { std::cos(2 * M_PI / n * k * b), -std::sin(2 * M_PI / n * k * b) };
    dst[id2(j, k + 0, a * 2)] = src[id2(j, k, a)] + src[id2(j + b, k, a)] * w;
    dst[id2(j, k + a, a * 2)] = src[id2(j, k, a)] - src[id2(j + b, k, a)] * w;
  }

  auto fft_stockham_for(std::vector<comp_t>& input) {
    std::vector<comp_t> buffer(input.size());

    int n = input.size();
    int p = log2(n);

    int a = 1;
    int b = n >> 1;

    for (int l = 0; l < p; l++) {
      for (int k = 0; k < a; k++) {
        for (int j = 0; j < b; j++) {
          if (l % 2 == 0)
            stockham_butterfly(input, buffer, j, k, a, b, n);
          else
            stockham_butterfly(buffer, input, j, k, a, b, n);
        }
      }
      a <<= 1;
      b >>= 1;
    }

    if (p % 2 == 0)
      return input;
    else
      return buffer;
  }
} // namespace conv