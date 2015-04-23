#pragma once

// Get current time (measured in milliseconds).
inline double currentTime()
{
	auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
	int freq = std::chrono::high_resolution_clock::duration::period::den;
	// Assuming period has numerator 1
	return (double)t.count() / (freq / 1000);
}
