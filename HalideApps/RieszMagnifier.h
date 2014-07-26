#pragma once

#include "VideoApp.h"

class RieszMagnifier
{
public:
	RieszMagnifier(VideoApp app, int pyramidLevels = 5);
	void process(Halide::Image<float> frame, Halide::Image<float> out);

private:
	const int CIRCBUFFER_SIZE = 5;

	VideoApp app;
	int pyramidLevels;
	std::vector<int> alphaValues;

	Halide::ImageParam input;
	Halide::Param<int> pParam;
	Halide::Func output;

	std::vector<Halide::Image<float>> pyramidBuffer;
	std::vector<Halide::Image<float>> temporalOutBuffer;

	int frameCounter;
};
