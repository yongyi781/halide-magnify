#pragma once

class RieszMagnifier
{
public:
	RieszMagnifier(int channels, Halide::Type type, int pyramidLevels = 5);
	void compileJIT(bool tile, Halide::Target target = Halide::get_target_from_environment());
	void compileToFile(std::string filenamePrefix, bool tile, Halide::Target target = Halide::get_target_from_environment());
	void bindJIT(float a1, float a2, float b0, float b1, float b2, float alpha, std::vector<Halide::Image<float>> historyBuffer);
	void process(Halide::Buffer frame, Halide::Buffer out);
	void computeBandSigmas();

	int getPyramidLevels() { return pyramidLevels; }

private:
	void schedule(bool tile, Halide::Target target = Halide::get_target_from_environment());
	void scheduleX86(bool tile);
	void scheduleARM(bool tile);

	static const int CIRCBUFFER_SIZE = 2;
	static const int NUM_BUFFER_TYPES = 7;

	// Spatial regularization
	std::vector<float> bandSigma;

	int channels;
	int pyramidLevels;

	// Input params
	Halide::ImageParam input;
	// Filter coefficients
	Halide::Param<float> a1{ "a1" }, a2{ "a2" }, b0{ "b0" }, b1{ "b1" }, b2{ "b2" };
	// Amplification coefficients
	Halide::Param<float> alpha{ "alpha" };
	// 4-dimensional buffer: For each pyramid level, an image of size width x height x type x circular buffer index.
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
	Halide::Param<int> pParam{ "pParam" };

	// Funcs
	Halide::Var x{ "x" }, y{ "y" }, c{ "c" }, p{ "p" }, xi{ "xi" }, yi{ "yi" };
	std::vector<Halide::Func>
		gPyramidDownX,
		gPyramid,
		lPyramidUpX,
		lPyramid,
		lPyramidCopy,
		clampedPyramidBuffer,
		r1Pyramid,
		r1Prev,
		r2Pyramid,
		r2Prev,
		productReal,
		productI,
		productJ,
		ijAmplitude,
		amplitude,
		phi,
		qPhaseDiffC,
		qPhaseDiffS,
		phaseC,
		phaseS,
		phaseCCopy,
		phaseSCopy,
		changeC,
		lowpass1C,
		lowpass2C,
		changeS,
		lowpass1S,
		lowpass2S,
		lowpass1CCopy,
		lowpass2CCopy,
		lowpass1SCopy,
		lowpass2SCopy,
		changeC2,
		changeS2,
		amp,
		changeCAmp,
		changeCRegX,
		changeCReg,
		changeSAmp,
		changeSRegX,
		changeSReg,
		ampRegX,
		ampReg,
		magC,
		pair,
		outLPyramid,
		outGPyramidUpX,
		outGPyramid;

	Halide::Func floatOutput{ "floatOutput" }, output{ "output" };

	int frameCounter;
};
