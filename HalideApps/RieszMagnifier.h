#pragma once

#include "VideoApp.h"

class RieszMagnifier
{
public:
	RieszMagnifier(int channels, Halide::Type type, int pyramidLevels = 5);
	void compileJIT(Halide::Target target = Halide::get_target_from_environment());
	void compileToFile(std::string filenamePrefix, Halide::Target target = Halide::get_target_from_environment());
	void bindJIT(float a1, float a2, float b0, float b1, float b2, float alpha, std::vector<Halide::Image<float>> historyBuffer);
	void process(Halide::Buffer frame, Halide::Buffer out);
	void computeBandSigmas();

	int getPyramidLevels() { return pyramidLevels; }

	static void computeFilter(double fps, double freqCenter, double freqWidth, std::vector<double>& filterA, std::vector<double>& filterB);

private:

	static const int CIRCBUFFER_SIZE = 2;
	static const int NUM_BUFFER_TYPES = 7;

	// Spatial regularization
	std::vector<float> bandSigma;

	int channels;
	int pyramidLevels;

	// Input params
	Halide::ImageParam input;
	// Filter coefficients
	Halide::Param<float> a1, a2, b0, b1, b2;
	// Amplification coefficients
	Halide::Param<float> alpha;
	// 4-dimensional buffer: For each pyramid level, an image of size width x height x circular buffer index x type.
	// Types:
	// ------
	// 0: pyramidBuffer
	// 1: phaseCBuffer
	// 2: phaseSBuffer
	// 3: lowpass1CBuffer
	// 4: lowpass2CBuffer
	// 5: lowpass1SBuffer
	// 6: lowpass2SBuffer
	std::vector<Halide::ImageParam> historyBuffer;
	// Current frame modulo 2. (For circular buffer).
	Halide::Param<int> pParam;

	Halide::Func output;

	int frameCounter;
};
