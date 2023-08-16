#pragma once

#include <vector>

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

} // namespace conv