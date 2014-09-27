#pragma once

#include "VideoApp.h"

class RieszMagnifier
{
public:
	RieszMagnifier(VideoApp app, int pyramidLevels = 5, double freqCenter = 2.0, double freqWidth = 0.5);
	void process(const Halide::Image<float>& frame, const Halide::Image<float>& out);
	void computeFilter();
	void computePerBandAlpha();

private:
	static const int CIRCBUFFER_SIZE = 2;

	// temporal filtering
	double freqCenter;
	double freqWidth;
	std::vector<double> filterA;
	std::vector<double> filterB;
	float alpha = 30;
	float lambdaC = 50;
	std::vector<float> bandAlphas;

	VideoApp app;
	int pyramidLevels;

	Halide::ImageParam input;
	Halide::Param<int> pParam;
	Halide::Func output;

	std::vector<Halide::Image<float>> pyramidBuffer;
	std::vector<Halide::Image<float>> r1Buffer;
	std::vector<Halide::Image<float>> r2Buffer;
	std::vector<Halide::Image<float>> phaseCBuffer;
	std::vector<Halide::Image<float>> phaseSBuffer;
	std::vector<Halide::Image<float>> lowpass1CBuffer;
	std::vector<Halide::Image<float>> lowpass2CBuffer;
	std::vector<Halide::Image<float>> lowpass1SBuffer;
	std::vector<Halide::Image<float>> lowpass2SBuffer;

	int frameCounter;
};
