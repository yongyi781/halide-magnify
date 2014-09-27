#pragma once

#include "VideoApp.h"

class EulerianMagnifier
{
public:
	EulerianMagnifier(VideoApp app, int pyramidLevels, const std::vector<float> alphaValues);
	void process(const Halide::Image<float>& frame, const Halide::Image<float>& out);

private:
	const static int CIRCBUFFER_SIZE = 5;

	VideoApp app;
	int pyramidLevels;

	Halide::ImageParam input;
	Halide::Param<int> pParam;
	Halide::Func output;

	std::vector<Halide::Image<float>> pyramidBuffer;
	std::vector<Halide::Image<float>> temporalOutBuffer;

	int frameCounter;
};
