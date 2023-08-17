#pragma once

#include <cassert>
#include <vector>
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>

using num_type = double;
using comp = std::complex<num_type>;

namespace conv {
	
	// all signals are represented as std::vector<float>

	void zero_pad(std::vector<float>& signal, size_t size) {
		auto original_size = signal.size();
		signal.resize(size);

		// if the target size is larger than the original size
		if (original_size < size) {
			for (int i = original_size; i < size; i++) {
				signal[i] = 0.f;
			}
		}
	}

	void fft_rec(
		std::vector<comp>& input,
		int stride,
		int first_index,
		int bit,
		int n) {
		if (n > 1) {
			assert(n % 2 == 0 && "fft : input size must be equal to 2^k.");
			const int h = n / 2;
			const num_type theta = 2.f * M_PI / static_cast<num_type>(n);

			for (int i = 0; i < h; i++) {
				const comp wi = { std::cos(i * theta), -sin(i * theta) };
				const comp a = input[first_index + i + 0];
				const comp b = input[first_index + i + h];

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
	std::vector<comp> fft(std::vector<comp>& input) {
		int n = input.size();

		fft_rec(input, 1, 0, 0, n);

		return input;
	}

	// ifft from freq to time
	std::vector<comp> ifft(std::vector<comp>& input) {
		int n = input.size();

		for (int i = 0; i < n; i++) {
			input[i] = conj(input[i]);
		}

		fft_rec(input, 1, 0, 0, n);

		for (int i = 0; i < n; i++) {
			input[i] = conj(input[i] / static_cast<num_type>(n));
		}

		return input;
	}

} // namespace conv