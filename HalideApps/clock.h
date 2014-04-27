#pragma once

#include <stdint.h>

extern "C" bool QueryPerformanceCounter(uint64_t *);
extern "C" bool QueryPerformanceFrequency(uint64_t *);

// Get current time (measured in milliseconds).
inline double currentTime()
{
	uint64_t t, freq;
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&freq);
	return (t * 1000.0) / freq;
}
